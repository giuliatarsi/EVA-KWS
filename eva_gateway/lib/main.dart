// ==============================================================================
// LIBRERIE UTILIZZATE E LORO FUNZIONE
// ==============================================================================
import 'dart:async'; // Importa classi per operazioni che richiedono tempo (Future) e flussi continui (Stream).
import 'dart:convert'; // Importa strumenti per convertire dati, come utf8.decode e jsonEncode.
import 'package:flutter/material.dart'; // Importa il pacchetto base di Flutter per creare l'interfaccia grafica (UI).
import 'package:flutter_blue_plus/flutter_blue_plus.dart'; // Importa la libreria per gestire il Bluetooth Low Energy (BLE).
import 'package:geolocator/geolocator.dart'; // Importa la libreria per leggere le coordinate del chip GPS del telefono.
import 'package:cloud_firestore/cloud_firestore.dart'; // Importa il client per connettersi al database Firebase Firestore.
import 'package:permission_handler/permission_handler.dart'; // Importa la libreria per mostrare i pop-up di richiesta permessi.
import 'package:flutter_background/flutter_background.dart'; // Importa la libreria per mantenere l'app attiva in background.
import 'package:firebase_core/firebase_core.dart'; // Importa il nucleo principale di Firebase per inizializzare il progetto.
import 'firebase_options.dart'; // Importa il file autogenerato con le chiavi segrete del tuo progetto Firebase.

// LIBRERIE PER LE NOTIFICHE E IL NUOVO BACKEND VERCEL
import 'package:firebase_messaging/firebase_messaging.dart'; // Importa la libreria per ricevere notifiche push tramite FCM.
import 'package:http/http.dart' as http; // Importa la libreria per effettuare chiamate web e la rinomina 'http' per comodità.

// ==============================================================================
// HANDLER NOTIFICHE IN BACKGROUND
// ==============================================================================
@pragma('vm:entry-point') // Forza il compilatore a mantenere questa funzione in memoria anche se l'app è chiusa.
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async { // Funzione asincrona che scatta quando arriva una notifica push in background.
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform); // Inizializza Firebase nel nuovo spazio di memoria isolato.
  print("🚨 Notifica ricevuta da Vercel in background: ${message.messageId}"); // Stampa nel terminale l'ID della notifica ricevuta.
}

// ==============================================================================
// PUNTO DI INGRESSO DELL'APPLICAZIONE
// ==============================================================================
void main() async { // La funzione 'main' è il punto di partenza dell'app; 'async' permette l'uso di 'await' al suo interno.
  WidgetsFlutterBinding.ensureInitialized(); // Assicura che il motore grafico di Flutter sia pronto prima di avviare plugin nativi.
  await Firebase.initializeApp( // Attende che l'inizializzazione del collegamento con i server di Firebase sia completata.
    options: DefaultFirebaseOptions.currentPlatform, // Passa le configurazioni specifiche per Android/iOS prese dal file delle chiavi.
  );
  runApp(const EmergencyApp()); // Avvia l'interfaccia grafica passando il widget principale 'EmergencyApp'.
}

class EmergencyApp extends StatelessWidget { // Crea un widget "stateless", ovvero una schermata che non cambia stato da sola.
  const EmergencyApp({super.key}); // Costruttore standard del widget.

  @override // Indica che stiamo sovrascrivendo il metodo 'build' della classe padre.
  Widget build(BuildContext context) { // Metodo che disegna materialmente questa parte di interfaccia.
    return const MaterialApp( // Ritorna un'applicazione in stile "Material Design" (standard Google).
      home: BleScannerPage(), // Imposta la pagina 'BleScannerPage' come schermata iniziale dell'app.
    );
  }
}

class BleScannerPage extends StatefulWidget { // Crea un widget "stateful", capace di aggiornarsi dinamicamente (es. quando si connette).
  const BleScannerPage({super.key}); // Costruttore standard del widget.

  @override // Sovrascrive il metodo per creare lo stato associato a questo widget.
  State<BleScannerPage> createState() => _BleScannerPageState(); // Collega questo widget alla classe di stato '_BleScannerPageState'.
}

class _BleScannerPageState extends State<BleScannerPage> { // Classe che contiene i dati e la logica della schermata Bluetooth.
  // ==============================================================================
  // VARIABILI GLOBALI DI STATO E FILTRI
  // ==============================================================================
  final String targetDeviceName = "SOS_Bracelet_S3"; // Salva il nome esatto del tuo dispositivo ESP32 per filtrarlo durante la scansione.
  final String serviceUuid = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"; // Salva l'ID univoco (UUID) del Servizio BLE creato sul tuo ESP32.
  final String charUuid = "beb5483e-36e1-4688-b7f5-ea07361b26a8"; // Salva l'UUID della Caratteristica BLE in cui l'ESP32 scriverà "SOS".
  
  final String myUserId = "utente_eva_01"; // Definisce un ID utente temporaneo per simulare un account loggato.

  BluetoothDevice? connectedDevice; // Variabile per memorizzare l'oggetto ESP32 una volta connesso. Il '?' indica che inizialmente può essere nullo.
  String statusText = "Disconnesso. Premi per cercare il dispositivo."; // Variabile di testo che viene mostrata al centro dello schermo.
  
  StreamSubscription<List<int>>? notifySubscription; // Variabile per memorizzare il "tubo" in ascolto dei dati inviati dal braccialetto.
  StreamSubscription<BluetoothConnectionState>? connectionSubscription; // Variabile per il "tubo" in ascolto dei cambi di stato (connesso/disconnesso).

  final String vercelEndpoint = "https://eva-kappa-beige.vercel.app/api/inviaNotifica"; // Salva l'indirizzo web (URL) della tua futura funzione Vercel.

  // ==============================================================================
  // INIZIALIZZAZIONE STATO E NOTIFICHE
  // ==============================================================================
  @override // Sovrascrive il metodo di inizializzazione base.
  void initState() { // Funzione chiamata automaticamente una sola volta quando la pagina viene creata.
    super.initState(); // Richiama il codice di inizializzazione della classe padre.
    _inizializzaNotifiche(); // Lancia immediatamente la funzione per configurare la ricezione delle notifiche FCM.
  }

  Future<void> _inizializzaNotifiche() async { // Funzione asincrona che gestisce permessi e token delle notifiche.
    FirebaseMessaging messaging = FirebaseMessaging.instance; // Crea un'istanza (un riferimento rapido) del servizio Firebase Cloud Messaging.
    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler); // Collega la funzione di background (definita sopra) per i messaggi a schermo spento.

    NotificationSettings settings = await messaging.requestPermission( // Fa apparire il pop-up di sistema chiedendo il permesso per le notifiche.
      alert: true, // Richiede il permesso di mostrare banner a schermo.
      badge: true, // Richiede il permesso di aggiornare il "pallino" rosso sull'icona dell'app.
      sound: true, // Richiede il permesso di riprodurre un suono all'arrivo della notifica.
    );

    if (settings.authorizationStatus == AuthorizationStatus.authorized) { // Controlla se l'utente ha premuto "Consenti" sul pop-up.
      String? token = await messaging.getToken(); // Richiede a Firebase il token FCM univoco di questo specifico telefono fisico.
      if (token != null) { // Se Firebase ha effettivamente restituito un token valido...
        await _salvaTokenFCM(token); // ...chiama la funzione per salvare questo token nel database Firestore.
      }
      messaging.onTokenRefresh.listen((newToken) { // Si mette in ascolto: se Firebase decide di rigenerare il token per sicurezza...
        _salvaTokenFCM(newToken); // ...salva automaticamente il nuovo token aggiornato nel database.
      });
    }
  }

  Future<void> _salvaTokenFCM(String token) async { // Funzione per scrivere il token FCM su Firestore.
    await FirebaseFirestore.instance.collection('utenti').doc(myUserId).set({ // Va nella collezione 'utenti', documento 'utente_eva_01', e imposta i dati.
      'fcmToken': token, // Crea o aggiorna il campo chiamato 'fcmToken' inserendovi il token effettivo.
    }, SetOptions(merge: true)); // L'opzione 'merge: true' fa sì che se il documento contiene altri campi (es. nome utente), questi non vengano cancellati.
  }

  // ==============================================================================
  // 1. RICHIESTA PERMESSI E CONFIGURAZIONE BACKGROUND
  // ==============================================================================
  Future<bool> setupPermissionsAndBackground() async { // Funzione che controlla i permessi hardware prima della scansione BLE.
    setState(() => statusText = "Richiesta permessi in corso..."); // Aggiorna il testo a schermo per informare l'utente del processo in corso.

    Map<Permission, PermissionStatus> statuses = await [ // Crea una lista di permessi da richiedere al sistema operativo.
      Permission.location, // Permesso per accedere alla posizione (necessario per il GPS e per la scansione Bluetooth su Android).
      Permission.bluetoothScan, // Permesso specifico per attivare il radar Bluetooth.
      Permission.bluetoothConnect, // Permesso specifico per stabilire una connessione Bluetooth.
    ].request(); // Lancia fisicamente i pop-up di richiesta permessi e attende le risposte.

    if (statuses[Permission.bluetoothScan] != PermissionStatus.granted || // Se il permesso di scansione BLE è stato negato...
        statuses[Permission.bluetoothConnect] != PermissionStatus.granted) { // ...OPPURE se il permesso di connessione BLE è stato negato...
      setState(() => statusText = "Errore: Permessi Bluetooth negati!"); // ...aggiorna il testo a schermo mostrando un errore.
      return false; // Interrompe la funzione e restituisce 'falso', impedendo all'app di procedere.
    }

    const androidConfig = FlutterBackgroundAndroidConfig( // Prepara la configurazione per il servizio che terrà l'app viva in background.
      notificationTitle: "Gateway Emergenza Attivo", // Definisce il titolo della notifica ineliminabile che apparirà nel menu a tendina.
      notificationText: "In ascolto del braccialetto in background", // Definisce il testo descrittivo di tale notifica.
      notificationImportance: AndroidNotificationImportance.normal, // Imposta la priorità normale per non essere troppo invasiva.
    );
    
    bool hasBackgroundPermissions = await FlutterBackground.initialize(androidConfig: androidConfig); // Inizializza il plugin passando la configurazione appena creata.
    if (hasBackgroundPermissions) { // Se l'inizializzazione va a buon fine (il telefono lo supporta)...
      await FlutterBackground.enableBackgroundExecution(); // ...attiva fisicamente l'esecuzione in background forzata.
    }
    return true; // Restituisce 'vero', comunicando che si può procedere con sicurezza alla scansione BLE.
  }

  // ==============================================================================
  // 2 & 3. SCANSIONE, CONNESSIONE E ASCOLTO BLE
  // ==============================================================================
  Future<void> scanAndConnect() async { // Funzione principale per cercare e agganciare l'ESP32.
    bool canProceed = await setupPermissionsAndBackground(); // Richiama il blocco dei permessi e aspetta l'esito.
    if (!canProceed) return; // Se l'esito è falso (permessi negati), interrompe immediatamente la scansione.

    setState(() => statusText = "Ricerca di $targetDeviceName in corso..."); // Mostra a schermo che il radar è acceso.
    await FlutterBluePlus.startScan(timeout: const Duration(seconds: 10)); // Accende il radar Bluetooth per un massimo di 10 secondi per risparmiare batteria.

    FlutterBluePlus.scanResults.listen((results) async { // Apre un canale in ascolto sui dispositivi (results) man mano che vengono trovati.
      for (ScanResult r in results) { // Inizia un ciclo per analizzare singolarmente ogni dispositivo trovato nella lista.
        if (r.device.platformName == targetDeviceName) { // Controlla se il nome del dispositivo trovato combacia col nome dell'ESP32 salvato in alto.
          await FlutterBluePlus.stopScan(); // Siccome abbiamo trovato il dispositivo giusto, spegniamo subito il radar.
          setState(() => statusText = "Dispositivo trovato! Connessione..."); // Aggiorna il testo a schermo informando l'utente.
          await connectToDevice(r.device); // Lancia la funzione di connessione passandole lo specifico dispositivo appena trovato.
          break; // Ferma il ciclo 'for', evitando di controllare altri dispositivi inutilmente.
        }
      }
    });
  }

  Future<void> connectToDevice(BluetoothDevice device) async { // Funzione che gestisce l'aggancio e l'ascolto dei sensori.
    await device.connect(autoConnect: false); // Comanda fisicamente al telefono di connettersi al dispositivo. (autoConnect: false evita bug su Android).
    setState(() { // Notifica a Flutter che dei dati sono cambiati.
      connectedDevice = device; // Salva il dispositivo connesso nella variabile globale.
      statusText = "Connesso! Configurazione ascolto SOS..."; // Aggiorna il testo a schermo.
    });

    connectionSubscription = device.connectionState.listen((BluetoothConnectionState state) { // Apre il canale in ascolto per monitorare se la connessione cade.
      if (state == BluetoothConnectionState.disconnected) { // Se lo stato passa da connesso a "disconnesso" (es. braccialetto spento o lontano)...
        setState(() { // Avvisa l'interfaccia del cambiamento.
          connectedDevice = null; // Svuota la variabile globale, poiché non siamo più connessi.
          statusText = "Dispositivo disconnesso! Riconnessione..."; // Aggiorna il testo a schermo.
        });
        scanAndConnect(); // Richiama automaticamente la funzione di scansione per tentare di riagganciarsi in loop.
      }
    });

    List<BluetoothService> services = await device.discoverServices(); // Interroga l'ESP32 per farsi restituire la lista dei "Servizi" che offre.
    for (var service in services) { // Avvia un ciclo per analizzare ogni servizio trovato.
      if (service.uuid.toString().toLowerCase() == serviceUuid) { // Se l'UUID del servizio analizzato corrisponde all'UUID del tuo progetto EVA...
        for (var characteristic in service.characteristics) { // ...avvia un ciclo per analizzare le "Caratteristiche" dentro questo Servizio.
          if (characteristic.uuid.toString().toLowerCase() == charUuid) { // Se l'UUID della caratteristica corrisponde a quella dedicata all'SOS...
            await characteristic.setNotifyValue(true); // ...attiva le notifiche push BLE (dice al telefono: "avvisami quando l'ESP32 scrive qui").
            
            notifySubscription = characteristic.onValueReceived.listen((value) { // Apre il canale in ascolto sui dati in arrivo dalla caratteristica.
              String message = utf8.decode(value); // Prende i byte grezzi in arrivo ('value') e li traduce in una stringa di testo usando la codifica UTF-8.
              if (message.trim() == "SOS") { // Se il testo (ripulito da eventuali spazi vuoti iniziali/finali) è esattamente la parola "SOS"...
                handleEmergencyAlert(); // ...allora scatena la funzione che gestisce l'emergenza (GPS e Firebase).
              }
            });
            setState(() => statusText = "Protetto: In ascolto di segnali SOS."); // Aggiorna a schermo informando che il sistema è pienamente armato.
          }
        }
      }
    }
  }

  // ==============================================================================
  // 4. GESTIONE EMERGENZA: LETTURA GPS
  // ==============================================================================
  Future<void> handleEmergencyAlert() async { // Funzione chiamata quando si preme il pulsante sul braccialetto (o il pulsante rosso di test).
    setState(() => statusText = "ALLARME RICEVUTO! Recupero posizione GPS..."); // Aggiorna il testo a schermo.

    try { // Avvia un blocco "try" per tentare di leggere il GPS, catturando eventuali errori (es. GPS spento).
      Position position = await Geolocator.getCurrentPosition( // Interroga il chip hardware del GPS.
        desiredAccuracy: LocationAccuracy.high, // Richiede la precisione massima disponibile (fondamentale per le emergenze, consuma più batteria).
      );
      String mapsLink = "https://www.google.com/maps?q=${position.latitude},${position.longitude}"; // Incolla latitudine e longitudine in un indirizzo web creando un link Google Maps cliccabile.
      await sendAlertToContacts(mapsLink); // Passa questo link alla funzione che contatterà il server.
      
      setState(() => statusText = "SOS inviato con successo tramite Vercel!"); // Mostra il messaggio di successo finale.
    } catch (e) { // Se qualcosa va storto nella lettura del GPS (il blocco 'try' fallisce)...
      setState(() => statusText = "Errore durante l'invio dell'SOS: $e"); // ...stampa a schermo il motivo tecnico dell'errore (la variabile 'e').
    }
  }

  // ==============================================================================
  // 5. INVIO ALLARME A VERCEL E FIRESTORE
  // ==============================================================================
  Future<void> sendAlertToContacts(String mapsLink) async { // Funzione per comunicare con il database e il server, ricevendo in input il link della mappa.
    try { // Blocco di sicurezza per catturare eventuali crolli di rete o mancanza di internet.
      // 1. Salviamo comunque lo storico su Firestore (gratuito) come registro degli eventi.
      await FirebaseFirestore.instance.collection('allarmi_emergenza').add({ // Aggiunge un nuovo documento alla collezione 'allarmi_emergenza'.
        'mittente': myUserId, // Salva chi ha lanciato l'allarme.
        'link_maps': mapsLink, // Salva il link generato poco prima.
        'timestamp': FieldValue.serverTimestamp(), // Usa l'orologio interno dei server Google per garantire l'esattezza dell'ora.
        'stato': 'attivo' // Imposta lo stato iniziale dell'emergenza.
      });
      
      // 2. Comunichiamo direttamente a Vercel di far partire la notifica push.
      Map<String, String> payload = { // Crea un "dizionario" (una mappa chiave-valore) contenente i dati da spedire a Vercel.
        'mittente': myUserId, // Passa l'ID dell'utente.
        'link_maps': mapsLink, // Passa il link della mappa.
      };

      final response = await http.post( // Esegue una richiesta web di tipo "POST" (per inviare dati) al server.
        Uri.parse(vercelEndpoint), // Converte l'URL di Vercel (definito in alto) nel formato formale richiesto dalla funzione.
        headers: {"Content-Type": "application/json"}, // Specifica al server Vercel che i dati che stiamo inviando sono in formato JSON.
        body: jsonEncode(payload), // Trasforma la mappa 'payload' in una vera stringa JSON per la trasmissione.
      );

      if (response.statusCode == 200) { // Il codice HTTP 200 significa "OK/Successo". Se Vercel risponde così...
        print("Allarme elaborato con successo da Vercel!"); // ...stampa in console la conferma.
      } else { // Se il server risponde con codici di errore (es. 404, 500)...
        print("Errore Vercel: ${response.body}"); // ...stampa in console il testo dell'errore restituito da Vercel.
        setState(() => statusText = "Il server Vercel ha restituito un errore."); // ...e avvisa l'utente tramite l'interfaccia.
      }
      
    } catch (e) { // Se l'app non riesce proprio a contattare internet...
      print("Errore di rete generale: $e"); // ...stampa l'eccezione nel terminale di debug.
      setState(() => statusText = "Errore di rete. Impossibile contattare Vercel."); // ...e informa l'utente a schermo.
    }
  }

  // ==============================================================================
  // 6. PULIZIA DELLA MEMORIA
  // ==============================================================================
  @override // Sovrascrive il metodo di distruzione base del widget.
  void dispose() { // Funzione che scatta quando il widget viene rimosso definitivamente (es. chiusura app).
    notifySubscription?.cancel(); // Chiude la tubatura in ascolto dei dati dell'ESP32 per evitare sprechi di RAM (memory leak).
    connectionSubscription?.cancel(); // Chiude la tubatura in ascolto dello stato del Bluetooth.
    connectedDevice?.disconnect(); // Ordina esplicitamente al telefono di sganciarsi dall'ESP32.
    FlutterBackground.disableBackgroundExecution(); // Spegne il servizio in background forzato, rimuovendo la notifica dal menu a tendina.
    super.dispose(); // Richiama il dispose della classe padre per concludere la distruzione del widget.
  }

  // ==============================================================================
  // 7. COSTRUZIONE INTERFACCIA VISIVA
  // ==============================================================================
  @override // Sovrascrive il metodo di disegno grafico.
  Widget build(BuildContext context) { // Funzione che viene richiamata ogni volta che fai 'setState' per ridisegnare lo schermo.
    return Scaffold( // Ritorna la struttura base di una pagina Flutter (incluso spazio per AppBar e Corpo).
      appBar: AppBar(title: const Text("Gateway Emergenza BLE")), // Disegna la barra superiore dell'app col titolo.
      body: Center( // Avvolge il contenuto principale, centrandolo orizzontalmente e verticalmente.
        child: Padding( // Aggiunge un margine interno attorno ai contenuti.
          padding: const EdgeInsets.all(24.0), // Definisce un margine uniforme di 24 pixel su tutti e quattro i lati.
          child: Column( // Impila i widget figli in verticale dall'alto verso il basso.
            mainAxisAlignment: MainAxisAlignment.center, // Centra verticalmente gli elementi all'interno della Column.
            children: [ // Inizia la lista dei widget visualizzati a schermo.
              Icon( // Disegna un'icona di sistema.
                connectedDevice != null ? Icons.bluetooth_connected : Icons.bluetooth_disabled, // Selettore logico: se connesso mostra l'icona "collegato", sennò l'icona "disabilitato".
                size: 80, // Imposta la grandezza dell'icona a 80 pixel.
                color: connectedDevice != null ? Colors.blue : Colors.grey, // Selettore logico per il colore: blu se connesso, grigio se disconnesso.
              ),
              const SizedBox(height: 20), // Disegna una scatola trasparente alta 20 pixel per creare spazio vuoto tra l'icona e il testo.
              Text(statusText, textAlign: TextAlign.center, style: const TextStyle(fontSize: 18)), // Stampa la variabile di stato dinamica 'statusText' centrata e con font 18.
              const SizedBox(height: 40), // Crea un altro spazio vuoto di 40 pixel prima dei pulsanti.
              ElevatedButton( // Disegna un bottone classico con sfondo pieno e lieve ombra.
                onPressed: connectedDevice == null ? scanAndConnect : null, // Selettore logico: bottone cliccabile (lancia scanAndConnect) solo se NON si è già connessi. Se connesso, diventa inattivo (null).
                child: const Text("Connetti e Attiva Protezione"), // Definisce il testo scritto dentro il bottone.
              ),
              const SizedBox(height: 20), // Altro spazio vuoto di 20 pixel.
              ElevatedButton( // Disegna il secondo bottone per testare le emergenze.
                style: ElevatedButton.styleFrom(backgroundColor: Colors.red), // Sovrascrive lo stile base per colorare questo bottone di rosso (colore di emergenza).
                onPressed: handleEmergencyAlert, // Collega direttamente il clic a 'handleEmergencyAlert' (bypassando il Bluetooth).
                child: const Text("TEST: Simula Allarme", style: TextStyle(color: Colors.white)), // Testo del bottone, forzato bianco per leggibilità sul rosso.
              ),
            ],
          ),
        ),
      ),
    );
  }
}