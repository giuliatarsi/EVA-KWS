// ==============================================================================
// LIBRERIE UTILIZZATE E LORO FUNZIONE
// ==============================================================================
import 'dart:async'; // Fornisce le classi Stream e Future per gestire le operazioni che richiedono tempo (es. attese di rete o Bluetooth).
import 'dart:convert'; // Contiene la funzione utf8.decode() per trasformare i byte ricevuti dal Bluetooth in testo leggibile.
import 'package:flutter/material.dart'; // Libreria principale di Flutter per disegnare l'interfaccia grafica (pulsanti, testi, icone).
import 'package:flutter_blue_plus/flutter_blue_plus.dart'; // Motore BLE: gestisce la scansione, connessione e lettura del braccialetto.
import 'package:geolocator/geolocator.dart'; // Interagisce con il chip GPS del telefono per ottenere le coordinate esatte.
import 'package:http/http.dart' as http; // Esegue le chiamate di rete (Internet) verso le API esterne (Telegram).

// NUOVE LIBRERIE AGGIUNTE PER I REQUISITI ARCHITETTURALI
import 'package:permission_handler/permission_handler.dart'; // Mostra all'utente i pop-up nativi per autorizzare l'uso di Bluetooth e GPS.
import 'package:flutter_background/flutter_background.dart'; // Mantiene l'app attiva in background (a schermo spento) creando un Foreground Service su Android.

// ==============================================================================
// PUNTO DI INGRESSO DELL'APPLICAZIONE
// ==============================================================================
void main() => runApp(const EmergencyApp());

class EmergencyApp extends StatelessWidget {
  const EmergencyApp({super.key});

  @override
  Widget build(BuildContext context) {
    // MaterialApp è il contenitore principale che gestisce la navigazione e il tema.
    return const MaterialApp(
      home: BleScannerPage(),
    );
  }
}

class BleScannerPage extends StatefulWidget {
  const BleScannerPage({super.key});

  @override
  State<BleScannerPage> createState() => _BleScannerPageState();
}

class _BleScannerPageState extends State<BleScannerPage> {
  // ==============================================================================
  // VARIABILI GLOBALI DI STATO
  // ==============================================================================
  // Questi parametri (UUID) dicono all'app esattamente quale dispositivo e servizio cercare.
  final String targetDeviceName = "SOS_Bracelet_S3";
  final String serviceUuid = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
  final String charUuid = "beb5483e-36e1-4688-b7f5-ea07361b26a8";

  BluetoothDevice? connectedDevice; // Memorizza il dispositivo quando è connesso.
  String statusText = "Disconnesso. Premi per cercare il dispositivo."; // Testo mostrato a schermo.
  
  StreamSubscription<List<int>>? notifySubscription; // Canale di ascolto per le notifiche "SOS".
  StreamSubscription<BluetoothConnectionState>? connectionSubscription; // NUOVO: Canale di ascolto per intercettare le disconnessioni.

  // ==============================================================================
  // 1. RICHIESTA PERMESSI E CONFIGURAZIONE BACKGROUND (NOVITÀ)
  // Scopo: Assicura che l'app abbia i diritti legali per usare l'hardware e per 
  // rimanere attiva a schermo spento prima di iniziare qualsiasi operazione BLE.
  // ==============================================================================
  Future<bool> setupPermissionsAndBackground() async {
    setState(() => statusText = "Richiesta permessi in corso...");

    // Richiede contemporaneamente permessi per Posizione e Bluetooth.
    Map<Permission, PermissionStatus> statuses = await [
      Permission.location,
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
    ].request();

    // Trasformazione dati: verifica che tutti i permessi critici siano stati concessi (isGranted).
    if (statuses[Permission.bluetoothScan] != PermissionStatus.granted ||
        statuses[Permission.bluetoothConnect] != PermissionStatus.granted) {
      setState(() => statusText = "Errore: Permessi Bluetooth negati!");
      return false;
    }

    // Configura il servizio in background (Foreground Service per Android).
    // Parametri passati: titolo e testo della notifica fissa che appare nel menu a tendina del telefono.
    const androidConfig = FlutterBackgroundAndroidConfig(
      notificationTitle: "Gateway Emergenza Attivo",
      notificationText: "In ascolto del braccialetto in background",
      notificationImportance: AndroidNotificationImportance.normal,
    );
    
    // Inizializza e attiva l'esecuzione in background.
    bool hasBackgroundPermissions = await FlutterBackground.initialize(androidConfig: androidConfig);
    if (hasBackgroundPermissions) {
      await FlutterBackground.enableBackgroundExecution();
    }
    
    return true; // Ritorna "vero" se l'app è autorizzata a procedere.
  }

  // ==============================================================================
  // 2. SCANSIONE AL DISPOSITIVO (MODIFICATO)
  // Scopo: Attiva il radar Bluetooth, filtra i risultati in base al nome e si aggancia.
  // ==============================================================================
  Future<void> scanAndConnect() async {
    // Prima di scansionare, esegue il blocco dei permessi. Se fallisce, interrompe (return).
    bool canProceed = await setupPermissionsAndBackground();
    if (!canProceed) return;

    setState(() => statusText = "Ricerca di $targetDeviceName in corso...");

    // Avvia la scansione per 10 secondi.
    await FlutterBluePlus.startScan(timeout: const Duration(seconds: 10));

    // Ascolta i risultati della scansione man mano che arrivano.
    FlutterBluePlus.scanResults.listen((results) async {
      for (ScanResult r in results) {
        // Se trova un dispositivo con il nome esatto, blocca la scansione e passa alla connessione.
        if (r.device.platformName == targetDeviceName) {
          await FlutterBluePlus.stopScan();
          setState(() => statusText = "Dispositivo trovato! Connessione...");
          
          await connectToDevice(r.device);
          break; // Ferma il ciclo for.
        }
      }
    });
  }

  // ==============================================================================
  // 3. CONNESSIONE, ASCOLTO SOS E RICONNESSIONE AUTOMATICA (MODIFICATO)
  // Scopo: Entra nel servizio GATT, attiva il campanello delle notifiche e 
  // riavvia automaticamente la ricerca se il braccialetto va fuori portata.
  // ==============================================================================
  Future<void> connectToDevice(BluetoothDevice device) async {
    await device.connect(autoConnect: false);
    setState(() {
      connectedDevice = device;
      statusText = "Connesso! Configurazione ascolto SOS...";
    });

    // NUOVO: LOGICA DI RICONNESSIONE AUTOMATICA
    // Ascolta lo stato del Bluetooth. Se il braccialetto si spegne o si allontana,
    // intercetta l'evento "disconnected" e richiama da solo scanAndConnect().
    connectionSubscription = device.connectionState.listen((BluetoothConnectionState state) {
      if (state == BluetoothConnectionState.disconnected) {
        setState(() {
          connectedDevice = null;
          statusText = "Dispositivo disconnesso! Tentativo di riconnessione...";
        });
        scanAndConnect(); // Riavvia autonomamente il flusso.
      }
    });

    // Cerca tutti i servizi offerti dal braccialetto.
    List<BluetoothService> services = await device.discoverServices();
    for (var service in services) {
      if (service.uuid.toString().toLowerCase() == serviceUuid) {
        for (var characteristic in service.characteristics) {
          if (characteristic.uuid.toString().toLowerCase() == charUuid) {
            
            // Abilita la ricezione delle notifiche sull'ESP32.
            await characteristic.setNotifyValue(true);
            
            // Ascolta i dati grezzi in ingresso.
            notifySubscription = characteristic.onValueReceived.listen((value) {
              // Trasformazione dati: converte i byte [83, 79, 83] nella stringa di testo "SOS".
              String message = utf8.decode(value);
              if (message.trim() == "SOS") {
                handleEmergencyAlert(); // Chiama la funzione di emergenza se riceve il trigger.
              }
            });

            setState(() => statusText = "Protetto: In ascolto di segnali SOS in background.");
          }
        }
      }
    }
  }

  // ==============================================================================
  // 4. GESTIONE EMERGENZA: LETTURA GPS
  // Scopo: Acquisire latitudine e longitudine dal chip hardware e creare un URL.
  // ==============================================================================
  Future<void> handleEmergencyAlert() async {
    setState(() => statusText = "ALLARME RICEVUTO! Recupero posizione GPS...");

    try {
      // Ottieni le coordinate attuali impostando la massima precisione (usa più batteria ma è salvavita).
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      // Trasformazione dati: concatena i numeri di latitudine e longitudine in un link Google Maps cliccabile.
      String mapsLink = "https://www.google.com/maps?q=${position.latitude},${position.longitude}";
      
      // Invia la posizione formattata ai contatti.
      await sendAlertToContacts(mapsLink);
      
      setState(() => statusText = "SOS e posizione inviati con successo!");
    } catch (e) {
      setState(() => statusText = "Errore durante l'invio dell'SOS: $e");
    }
  }

  // ==============================================================================
  // 5. INVIO AUTOMATICO DEL MESSAGGIO VIA RETE
  // Scopo: Bypassare i blocchi nativi degli SMS usando un Bot Telegram tramite HTTP.
  // ==============================================================================
  Future<void> sendAlertToContacts(String mapsLink) async {
    // Placeholder mantenuti come richiesto per le configurazioni future.
    const String botToken = "IL_TUO_BOT_TOKEN_TELEGRAM";
    const String chatId = "ID_CHAT_CONTATTO_EMERGENZA"; 
    
    final String text = "🚨 ALLARME EMERGENZA! Il dispositivo ha rilevato una richiesta di aiuto.\n📍 Posizione attuale: $mapsLink";

    // Trasformazione dati: Uri.encodeComponent() converte spazi e caratteri speciali (come emoji o newline)
    // nel formato sicuro per gli indirizzi web (es. lo spazio diventa %20).
    final Uri url = Uri.parse(
      "https://api.telegram.org/bot$botToken/sendMessage?chat_id=$chatId&text=${Uri.encodeComponent(text)}"
    );

    // Esegue una richiesta di rete GET verso il server di Telegram.
    await http.get(url);
  }

  // ==============================================================================
  // 6. PULIZIA DELLA MEMORIA E INTERFACCIA VISIVA
  // Scopo: Chiudere i canali quando l'app viene distrutta e disegnare la schermata.
  // ==============================================================================
  @override
  void dispose() {
    // Chiude i canali di ascolto per prevenire Memory Leak (consumo anomalo di RAM).
    notifySubscription?.cancel();
    connectionSubscription?.cancel();
    connectedDevice?.disconnect();
    
    // Disabilita il servizio in background quando l'app viene forzatamente chiusa dall'utente.
    FlutterBackground.disableBackgroundExecution();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Scaffold disegna la struttura base (Barra superiore e Corpo centrale).
    return Scaffold(
      appBar: AppBar(title: const Text("Gateway Emergenza BLE")),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                // Cambia il colore dell'icona dinamicamente in base allo stato della connessione.
                connectedDevice != null ? Icons.bluetooth_connected : Icons.bluetooth_disabled,
                size: 80,
                color: connectedDevice != null ? Colors.blue : Colors.grey,
              ),
              const SizedBox(height: 20),
              // Stampa a schermo la variabile statusText che abbiamo aggiornato in tutti i passaggi.
              Text(statusText, textAlign: TextAlign.center, style: const TextStyle(fontSize: 18)),
              const SizedBox(height: 40),
              ElevatedButton(
                // Se il dispositivo è già connesso, il pulsante viene disabilitato (passando null).
                onPressed: connectedDevice == null ? scanAndConnect : null,
                child: const Text("Connetti e Attiva Protezione"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}