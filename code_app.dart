import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;

void main() => runApp(const EmergencyApp());

class EmergencyApp extends StatelessWidget {
  const EmergencyApp({super.key});

  @override
  Widget build(BuildContext context) {
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
  // Gli stessi UUID definiti sull'ESP32-S3
  final String targetDeviceName = "SOS_Bracelet_S3";
  final String serviceUuid = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
  final String charUuid = "beb5483e-36e1-4688-b7f5-ea07361b26a8";

  BluetoothDevice? connectedDevice;
  String statusText = "Disconnesso. Premi per cercare il dispositivo.";
  StreamSubscription<List<int>>? notifySubscription;

  // 1. SCANSIONE E CONNESSIONE AL DISPOSITIVO
  Future<void> scanAndConnect() async {
    setState(() => statusText = "Ricerca di $targetDeviceName in corso...");

    // Avvia la scansione per 10 secondi
    await FlutterBluePlus.startScan(timeout: const Duration(seconds: 10));

    FlutterBluePlus.scanResults.listen((results) async {
      for (ScanResult r in results) {
        if (r.device.platformName == targetDeviceName) {
          await FlutterBluePlus.stopScan();
          setState(() => statusText = "Dispositivo trovato! Connessione...");
          
          await connectToDevice(r.device);
          break;
        }
      }
    });
  }

  // 2. SCOPERTA DEI SERVIZI E ISCRIZIONE ALLA NOTIFICA (Come in LightBlue)
  Future<void> connectToDevice(BluetoothDevice device) async {
    await device.connect(autoConnect: false);
    setState(() {
      connectedDevice = device;
      statusText = "Connesso! Configurazione ascolto SOS...";
    });

    List<BluetoothService> services = await device.discoverServices();
    for (var service in services) {
      if (service.uuid.toString().toLowerCase() == serviceUuid) {
        for (var characteristic in service.characteristics) {
          if (characteristic.uuid.toString().toLowerCase() == charUuid) {
            
            // Abilita la ricezione delle notifiche (Subscribe)
            await characteristic.setNotifyValue(true);
            
            // Ascolta il flusso di dati in arrivo dall'ESP32
            notifySubscription = characteristic.onValueReceived.listen((value) {
              String message = utf8.decode(value);
              if (message.trim() == "SOS") {
                handleEmergencyAlert();
              }
            });

            setState(() => statusText = "Protetto: In ascolto di segnali SOS.");
          }
        }
      }
    }
  }

  // 3. GESTIONE EMERGENZA: LETTURA GPS E INVIO NOTIFICA AI CONTATTI
  Future<void> handleEmergencyAlert() async {
    setState(() => statusText = "ALLARME RICEVUTO! Recupero posizione GPS...");

    try {
      // Controlla e richiedi i permessi GPS
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }

      // Ottieni le coordinate attuali ad alta precisione
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      String mapsLink = "https://www.google.com/maps?q=${position.latitude},${position.longitude}";
      
      // Invia la posizione ai contatti
      await sendAlertToContacts(mapsLink);
      
      setState(() => statusText = "SOS e posizione inviati con successo!");
    } catch (e) {
      setState(() => statusText = "Errore durante l'invio dell'SOS: $e");
    }
  }

  // 4. INVIO AUTOMATICO DEL MESSAGGIO (Esempio via Telegram Bot o Webhook/Twilio)
  Future<void> sendAlertToContacts(String mapsLink) async {
    // NOTA ARCHITETTURALE: Usare un Bot Telegram o un'API come Twilio permette 
    // l'invio silenzioso in background senza intervento manuale sul telefono.
    const String botToken = "IL_TUO_BOT_TOKEN_TELEGRAM";
    const String chatId = "ID_CHAT_CONTATTO_EMERGENZA"; 
    
    final String text = "🚨 ALLARME EMERGENZA! Il dispositivo ha rilevato una richiesta di aiuto.\n📍 Posizione attuale: $mapsLink";

    final Uri url = Uri.parse(
      "https://api.telegram.org/bot$botToken/sendMessage?chat_id=$chatId&text=${Uri.encodeComponent(text)}"
    );

    await http.get(url);
  }

  @override
  void dispose() {
    notifySubscription?.cancel();
    connectedDevice?.disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Gateway Emergenza BLE")),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                connectedDevice != null ? Icons.bluetooth_connected : Icons.bluetooth_disabled,
                size: 80,
                color: connectedDevice != null ? Colors.blue : Colors.grey,
              ),
              const SizedBox(height: 20),
              Text(statusText, textAlign: TextAlign.center, style: const TextStyle(fontSize: 18)),
              const SizedBox(height: 40),
              ElevatedButton(
                onPressed: connectedDevice == null ? scanAndConnect : null,
                child: const Text("Connetti al Dispositivo"),
              ),
            ],
          ),
        ),
      ),
    );
  }
}