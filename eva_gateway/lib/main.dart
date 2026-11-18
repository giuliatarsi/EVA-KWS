// ==============================================================================
// IMPORTAZIONE DELLE LIBRERIE
// Qui importiamo gli strumenti necessari: widget grafici, gestione Bluetooth,
// geolocalizzazione GPS, database Firestore, richieste HTTP e Autenticazione.
// ==============================================================================
import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_background/flutter_background.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';
import 'package:firebase_auth/firebase_auth.dart'; // Per gestire Login e Registrazione

// ==============================================================================
// HANDLER NOTIFICHE IN BACKGROUND
// Questa funzione speciale riceve le notifiche quando l'app è chiusa o in background.
// L'annotazione "@pragma('vm:entry-point')" serve a dire ad Android di non eliminare
// questa funzione durante l'ottimizzazione del codice.
// ==============================================================================
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  print("🚨 Notifica ricevuta da Vercel in background: ${message.messageId}");
}

// ==============================================================================
// FUNZIONE MAIN (PUNTO DI AVVIO)
// È la prima funzione che viene eseguita quando apri l'applicazione.
// Inizializza i servizi di Firebase e avvia l'app Flutter.
// ==============================================================================
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const EmergencyApp());
}

// ==============================================================================
// WIDGET PRINCIPALE DELL'APPLICAZIONE
// Configura lo stile visivo globale (Tema) e avvia il controllo dell'accesso.
// ==============================================================================
class EmergencyApp extends StatelessWidget {
  const EmergencyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EVA Emergency',
      theme: ThemeData(primarySwatch: Colors.red),
      // Controlla se l'utente ha già fatto il login o se deve farlo
      home: const AuthCheckWrapper(),
    );
  }
}

// ==============================================================================
// CONTROLLORE DI ACCESSO (AUTH CHECK)
// Verifica in tempo reale se l'utente è loggato:
// - Se SI -> Mostra la schermata principale Bluetooth (BleScannerPage).
// - Se NO -> Mostra la schermata di Login/Registrazione (LoginPage).
// ==============================================================================
class AuthCheckWrapper extends StatelessWidget {
  const AuthCheckWrapper({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.hasData && snapshot.data != null) {
          return const BleScannerPage();
        }
        return const LoginPage();
      },
    );
  }
}

// ==============================================================================
// SCHERMATA DI LOGIN / REGISTRAZIONE
// Permette all'utente di creare un account o entrare con Email e Password.
// ==============================================================================
class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  // Controller per leggere quello che l'utente scrive nei campi di testo
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  bool isLogin = true; // True = Schermata Login, False = Schermata Registrazione
  String errorMessage = "";

  // Funzione per inviare i dati a Firebase Auth
  Future<void> _submitAuth() async {
    try {
      setState(() => errorMessage = "");
      if (isLogin) {
        await FirebaseAuth.instance.signInWithEmailAndPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text.trim(),
        );
      } else {
        await FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: _emailController.text.trim(),
          password: _passwordController.text.trim(),
        );
      }
    } catch (e) {
      setState(() => errorMessage = e.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(isLogin ? "EVA - Accedi" : "EVA - Registrati")),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _emailController,
              decoration: const InputDecoration(labelText: "Email"),
              keyboardType: TextInputType.emailAddress,
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _passwordController,
              decoration: const InputDecoration(labelText: "Password"),
              obscureText: true, // Nasconde i caratteri della password
            ),
            const SizedBox(height: 20),
            if (errorMessage.isNotEmpty)
              Text(errorMessage, style: const TextStyle(color: Colors.red)),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _submitAuth,
              child: Text(isLogin ? "Accedi" : "Registrati"),
            ),
            TextButton(
              onPressed: () => setState(() => isLogin = !isLogin),
              child: Text(isLogin ? "Non hai un account? Registrati" : "Hai già un account? Accedi"),
            ),
          ],
        ),
      ),
    );
  }
}

// ==============================================================================
// SCHERMATA PRINCIPALE (BLUETOOTH, GPS E GESTIONE ALLARMI)
// ==============================================================================
class BleScannerPage extends StatefulWidget {
  const BleScannerPage({super.key});

  @override
  State<BleScannerPage> createState() => _BleScannerPageState();
}

class _BleScannerPageState extends State<BleScannerPage> {
  // Nome del dispositivo ESP32 e UUID del servizio Bluetooth BLE configurato nel firmware
  final String targetDeviceName = "SOS_Bracelet_S3";
  final String serviceUuid = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
  final String charUuid = "beb5483e-36e1-4688-b7f5-ea07361b26a8";
  
  // Identificativo univoco dell'utente basato sull'account Firebase loggato
  final String myUserId = FirebaseAuth.instance.currentUser!.uid; 

  BluetoothDevice? connectedDevice;
  String statusText = "Disconnesso. Premi per cercare il dispositivo.";
  
  StreamSubscription<List<int>>? notifySubscription;
  StreamSubscription<BluetoothConnectionState>? connectionSubscription;

  // L'indirizzo web del tuo backend serverless su Vercel
  final String vercelEndpoint = "https://eva-kappa-beige.vercel.app/api/inviaNotifica";

  @override
  void initState() {
    super.initState();
    _inizializzaNotifiche(); // Avvia l'ascolto delle notifiche all'avvio della schermata
  }

  // ==============================================================================
  // CONFIGURAZIONE DELLE NOTIFICHE E DEI LISTENER (FOREGROUND, BACKGROUND, CLICK)
  // ==============================================================================
  Future<void> _inizializzaNotifiche() async {
    FirebaseMessaging messaging = FirebaseMessaging.instance;
    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

    // Chiede i permessi per mostrare le notifiche all'utente
    NotificationSettings settings = await messaging.requestPermission(
      alert: true, badge: true, sound: true,
    );

    if (settings.authorizationStatus == AuthorizationStatus.authorized) {
      String? token = await messaging.getToken();
      if (token != null) {
        await _salvaTokenFCM(token);
      }
      messaging.onTokenRefresh.listen((newToken) {
        _salvaTokenFCM(newToken);
      });
    }

    // 1. LISTENER IN PRIMO PIANO (FOREGROUND): Se l'app è aperta e riceve un messaggio
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      print("🔔 Notifica ricevuta con app aperta!");
      _mostraPopupEmergenza(message); // Fa comparire il pop-up rosso
    });

    // 2. Se l'utente clicca la notifica mentre l'app è in BACKGROUND
    FirebaseMessaging.onMessageOpenedApp.listen((RemoteMessage message) {
      _apriMappaDaNotifica(message);
    });

    // 3. Se l'utente clicca la notifica e l'app era COMPLETAMENTE CHIUSA
    RemoteMessage? initialMessage = await messaging.getInitialMessage();
    if (initialMessage != null) {
      _apriMappaDaNotifica(initialMessage);
    }
  }

  // ==============================================================================
  // POP-UP DI EMERGENZA A SCHERMO (RICHIESTO AL PUNTO 3)
  // Mostra un banner rosso d'emergenza se l'app è attiva quando arriva l'allarme
  // ==============================================================================
  void _mostraPopupEmergenza(RemoteMessage message) {
    String titolo = message.notification?.title ?? "🚨 EMERGENZA EVA!";
    String corpo = message.notification?.body ?? "Richiesta di aiuto ricevuta.";
    String linkMaps = message.data['linkMaps'] ?? "";

    showDialog(
      context: context,
      barrierDismissible: false, // Impedisce di chiudere il pop-up toccando fuori dallo schermo
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: Colors.red[900], // Sfondo rosso scuro di allarme
          title: Text(
            titolo,
            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
          ),
          content: Text(
            corpo,
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
          actions: [
            // Se nel messaggio è presente il link GPS, mostra il pulsante per aprirlo
            if (linkMaps.isNotEmpty)
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(backgroundColor: Colors.white, foregroundColor: Colors.red),
                onPressed: () async {
                  Navigator.of(context).pop(); // Chiude il pop-up
                  final Uri url = Uri.parse(linkMaps);
                  await launchUrl(url, mode: LaunchMode.externalApplication); // Apre Google Maps
                },
                icon: const Icon(Icons.map),
                label: const Text("Apri Mappa GPS"),
              ),
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text("Chiudi", style: TextStyle(color: Colors.white70)),
            ),
          ],
        );
      },
    );
  }

  // Apre automaticamente la mappa quando viene cliccata la notifica di sistema
  Future<void> _apriMappaDaNotifica(RemoteMessage message) async {
    if (message.data.containsKey('linkMaps')) {
      String link = message.data['linkMaps'];
      if (link.isNotEmpty) {
        final Uri url = Uri.parse(link);
        try {
          await launchUrl(url, mode: LaunchMode.externalApplication);
        } catch (e) {
          print("Errore apertura mappa: $e");
        }
      }
    }
  }

  // Salva il Token univoco del dispositivo su Firebase Firestore
  Future<void> _salvaTokenFCM(String token) async {
    await FirebaseFirestore.instance.collection('utenti').doc(myUserId).set({
      'fcmToken': token,
      'email': FirebaseAuth.instance.currentUser?.email,
    }, SetOptions(merge: true));
  }

  // ==============================================================================
  // GESTIONE PERMESSI E ESECUZIONE IN BACKGROUND (ANDROID)
  // ==============================================================================
  Future<bool> setupPermissionsAndBackground() async {
    setState(() => statusText = "Richiesta permessi in corso...");

    // Richiede permessi di geolocalizzazione e bluetooth
    Map<Permission, PermissionStatus> statuses = await [
      Permission.location,
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
    ].request();

    if (statuses[Permission.bluetoothScan] != PermissionStatus.granted ||
        statuses[Permission.bluetoothConnect] != PermissionStatus.granted) {
      setState(() => statusText = "Errore: Permessi Bluetooth negati!");
      return false;
    }

    // Configura il servizio in background per non far chiudere l'app dal sistema Android
    const androidConfig = FlutterBackgroundAndroidConfig(
      notificationTitle: "Gateway Emergenza Attivo",
      notificationText: "In ascolto del braccialetto in background",
      notificationImportance: AndroidNotificationImportance.normal,
    );
    
    bool hasBackgroundPermissions = await FlutterBackground.initialize(androidConfig: androidConfig);
    if (hasBackgroundPermissions) {
      await FlutterBackground.enableBackgroundExecution();
    }
    return true;
  }

  // ==============================================================================
  // SCANSIONE E CONNESSIONE BLUETOOTH CON L'ESP32
  // ==============================================================================
  Future<void> scanAndConnect() async {
    bool canProceed = await setupPermissionsAndBackground();
    if (!canProceed) return;

    setState(() => statusText = "Ricerca di $targetDeviceName in corso...");
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

  // Connessione stabilita: si mette in ascolto della caratteristica BLE dell'allarme SOS
  Future<void> connectToDevice(BluetoothDevice device) async {
    await device.connect(autoConnect: false);
    setState(() {
      connectedDevice = device;
      statusText = "Connesso! Configurazione ascolto SOS...";
    });

    connectionSubscription = device.connectionState.listen((BluetoothConnectionState state) {
      if (state == BluetoothConnectionState.disconnected) {
        setState(() {
          connectedDevice = null;
          statusText = "Dispositivo disconnesso! Riconnessione...";
        });
        scanAndConnect();
      }
    });

    List<BluetoothService> services = await device.discoverServices();
    for (var service in services) {
      if (service.uuid.toString().toLowerCase() == serviceUuid) {
        for (var characteristic in service.characteristics) {
          if (characteristic.uuid.toString().toLowerCase() == charUuid) {
            await characteristic.setNotifyValue(true);
            
            // Ascolta i dati inviati via BLE dall'ESP32
            notifySubscription = characteristic.onValueReceived.listen((value) {
              String message = utf8.decode(value);
              if (message.trim() == "SOS") {
                handleEmergencyAlert(); // Se riceve "SOS", fa partire l'allarme!
              }
            });
            setState(() => statusText = "Protetto: In ascolto di segnali SOS.");
          }
        }
      }
    }
  }

  // ==============================================================================
  // GESTIONE DELL'EMERGENZA: GPS E CHIAMATA A VERCEL
  // ==============================================================================
  Future<void> handleEmergencyAlert() async {
    setState(() => statusText = "ALLARME RICEVUTO! Recupero posizione GPS...");

    try {
      // Cattura la posizione geografica attuale del telefono
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      String mapsLink = "https://www.google.com/maps?q=${position.latitude},${position.longitude}";
      
      // Invia i dati al database e chiama il server Vercel
      await sendAlertToContacts(mapsLink);
      
      setState(() => statusText = "SOS inviato con successo tramite Vercel!");
    } catch (e) {
      setState(() => statusText = "Errore durante l'invio dell'SOS: $e");
    }
  }

  Future<void> sendAlertToContacts(String mapsLink) async {
    try {
      // Salva lo storico dell'allarme su Firestore
      await FirebaseFirestore.instance.collection('allarmi_emergenza').add({
        'mittente': myUserId,
        'link_maps': mapsLink,
        'timestamp': FieldValue.serverTimestamp(),
        'stato': 'attivo'
      });
      
      Map<String, String> payload = {
        'mittente': myUserId,
        'link_maps': mapsLink,
      };

      // Invia la richiesta HTTP POST al backend Vercel per scatenare le notifiche push
      final response = await http.post(
        Uri.parse(vercelEndpoint),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        print("Allarme elaborato con successo da Vercel!");
      } else {
        setState(() => statusText = "Il server Vercel ha restituito un errore.");
      }
    } catch (e) {
      setState(() => statusText = "Errore di rete. Impossibile contattare Vercel.");
    }
  }

  // Pulisce le risorse e chiude le connessioni alla chiusura della pagina
  @override
  void dispose() {
    notifySubscription?.cancel();
    connectionSubscription?.cancel();
    connectedDevice?.disconnect();
    FlutterBackground.disableBackgroundExecution();
    super.dispose();
  }

  // ==============================================================================
  // INTERFACCIA GRAFICA (UI)
  // ==============================================================================
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Gateway Emergenza BLE"),
        actions: [
          // NUOVO: Pulsante per aprire la schermata dei Contatti di Emergenza
          IconButton(
            icon: const Icon(Icons.contacts),
            tooltip: "Contatti di Emergenza",
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const ContattiPage()),
              );
            },
          ),
          // Pulsante di Logout
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () => FirebaseAuth.instance.signOut(),
          )
        ],
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text("ID Utente: $myUserId", style: const TextStyle(fontSize: 12, color: Colors.grey)),
              const SizedBox(height: 10),
              Icon(
                connectedDevice != null ? Icons.bluetooth_connected : Icons.bluetooth_disabled,
                size: 80,
                color: connectedDevice != null ? Colors.blue : Colors.grey,
              ),
              const SizedBox(height: 20),
              Text(statusText, textAlign: TextAlign.center, style: const TextStyle(fontSize: 18)),
              const SizedBox(height: 40),
              // Pulsante per avviare la ricerca e la connessione BLE con l'ESP32
              ElevatedButton(
                onPressed: connectedDevice == null ? scanAndConnect : null,
                child: const Text("Connetti e Attiva Protezione"),
              ),
              const SizedBox(height: 20),
              // Pulsante di test per simulare un allarme manualmente
              ElevatedButton(
                style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                onPressed: handleEmergencyAlert,
                child: const Text("TEST: Simula Allarme", style: TextStyle(color: Colors.white)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
// ==============================================================================
// SCHERMATA GESTIONE CONTATTI DI EMERGENZA
// Permette di cercare un utente registrato tramite Email e aggiungerlo
// alla propria lista di soccorritori su Firestore.
// ==============================================================================
class ContattiPage extends StatefulWidget {
  const ContattiPage({super.key});

  @override
  State<ContattiPage> createState() => _ContattiPageState();
}

class _ContattiPageState extends State<ContattiPage> {
  final TextEditingController _emailController = TextEditingController();
  final String myUserId = FirebaseAuth.instance.currentUser!.uid;
  String feedbackMessage = "";

  // Cerca l'utente su Firestore tramite la sua email e lo salva nei contatti
  Future<void> _aggiungiContatto() async {
    String emailCercata = _emailController.text.trim();
    if (emailCercata.isEmpty) return;

    setState(() => feedbackMessage = "Ricerca in corso...");

    try {
      // 1. Cerca nella collezione 'utenti' chi ha quell'email
      QuerySnapshot query = await FirebaseFirestore.instance
          .collection('utenti')
          .where('email', isEqualTo: emailCercata)
          .get();

      if (query.docs.isEmpty) {
        setState(() => feedbackMessage = "Nessun utente registrato trovato con questa email.");
        return;
      }

      // 2. Recupera l'ID univoco (UID) del contatto trovato
      String contactUid = query.docs.first.id;

      // 3. Salva l'UID del contatto nella propria sottocollezione 'contatti_emergenza'
      await FirebaseFirestore.instance
          .collection('utenti')
          .doc(myUserId)
          .collection('contatti_emergenza')
          .doc(contactUid)
          .set({
        'email': emailCercata,
        'aggiuntoIl': FieldValue.serverTimestamp(),
      });

      _emailController.clear();
      setState(() => feedbackMessage = "Contatto aggiunto con successo!");
    } catch (e) {
      setState(() => feedbackMessage = "Errore: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Contatti di Emergenza")),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            // Campo per inserire l'email del contatto
            TextField(
              controller: _emailController,
              decoration: const InputDecoration(
                labelText: "Email del contatto registrato",
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.emailAddress,
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: _aggiungiContatto,
              icon: const Icon(Icons.person_add),
              label: const Text("Aggiungi Contatto"),
            ),
            if (feedbackMessage.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text(feedbackMessage, style: const TextStyle(fontWeight: FontWeight.bold)),
              ),
            const Divider(height: 40),
            const Text("I tuoi contatti attuali:", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            // Lista in tempo reale dei contatti salvati su Firestore
            Expanded(
              child: StreamBuilder<QuerySnapshot>(
                stream: FirebaseFirestore.instance
                    .collection('utenti')
                    .doc(myUserId)
                    .collection('contatti_emergenza')
                    .snapshots(),
                builder: (context, snapshot) {
                  if (!snapshot.hasData) return const Center(child: CircularProgressIndicator());
                  final docs = snapshot.data!.docs;
                  if (docs.isEmpty) return const Text("Nessun contatto aggiunto.");

                  return ListView.builder(
                    itemCount: docs.length,
                    itemBuilder: (context, index) {
                      var data = docs[index].data() as Map<String, dynamic>;
                      return ListTile(
                        leading: const Icon(Icons.person, color: Colors.red),
                        title: Text(data['email'] ?? docs[index].id),
                        trailing: IconButton(
                          icon: const Icon(Icons.delete, color: Colors.grey),
                          onPressed: () => docs[index].reference.delete(),
                        ),
                      );
                    },
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}