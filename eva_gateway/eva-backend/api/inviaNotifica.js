// Importa i moduli specifici seguendo le regole delle nuove versioni di Firebase
const { initializeApp, getApps, cert } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');
const { getMessaging } = require('firebase-admin/messaging');

module.exports = async (req, res) => {
  try {
    // Inizializzazione sicura con la nuova sintassi 'getApps()'
    if (getApps().length === 0) {
      if (!process.env.FIREBASE_CREDENTIALS) {
        return res.status(500).json({ error: "ERRORE VERCEL: La variabile FIREBASE_CREDENTIALS non esiste." });
      }
      
      let serviceAccount;
      try {
        serviceAccount = JSON.parse(process.env.FIREBASE_CREDENTIALS);
      } catch (e) {
        return res.status(500).json({ error: "ERRORE VERCEL: FIREBASE_CREDENTIALS non è un JSON valido." });
      }

      initializeApp({
        credential: cert(serviceAccount)
      });
    }

    // Estrae i dati inviati da Flutter
    const { mittente, link_maps } = req.body;
    if (!mittente) return res.status(400).json({ error: "ERRORE VERCEL: Nessun mittente fornito da Flutter." });

    // Usa il nuovo getFirestore() invece di admin.firestore()
    const db = getFirestore();
    const contattiSnapshot = await db.collection('utenti').doc(mittente).collection('contatti_emergenza').get();

    if (contattiSnapshot.empty) {
      return res.status(404).json({ error: "ERRORE DATABASE: Nessun contatto d'emergenza configurato per " + mittente });
    }

    const fcmTokens = [];
    for (const doc of contattiSnapshot.docs) {
      const utenteContatto = await db.collection('utenti').doc(doc.id).get();
      if (utenteContatto.data()?.fcmToken) fcmTokens.push(utenteContatto.data().fcmToken);
    }

    if (fcmTokens.length === 0) {
      return res.status(404).json({ error: "ERRORE FCM: I contatti esistono, ma nessuno ha registrato un token sul telefono." });
    }

    // Costruisce la notifica
    const payload = {
      notification: { 
        title: "🚨 EMERGENZA EVA!", 
        body: "Richiesta di aiuto. Tocca per aprire la posizione GPS." 
      },
      data: { 
        linkMaps: link_maps || "", 
        mittente: mittente 
      }
    };

    // Invia la notifica con il nuovo getMessaging()
    const response = await getMessaging().sendEachForMulticast({
      tokens: fcmTokens,
      notification: payload.notification,
      data: payload.data,
    });

    return res.status(200).json({ success: true, notificheInviate: response.successCount });

  } catch (error) {
    return res.status(500).json({ error: "CRASH INTERNO: " + error.message });
  }
};

/*const admin = require('firebase-admin');

// ==============================================================================
// INIZIALIZZAZIONE FIREBASE ADMIN SDK
// ==============================================================================
// Vercel riutilizza lo stesso spazio di memoria per chiamate ravvicinate.
// Questo controllo evita l'errore fatale di inizializzare Firebase due volte.
if (!admin.apps.length) {
  try {
    // process.env.FIREBASE_CREDENTIALS legge la chiave segreta (il file .json) 
    // che hai incollato nelle impostazioni web di Vercel.
    const serviceAccount = JSON.parse(process.env.FIREBASE_CREDENTIALS);
    
    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount)
    });
    console.log("Firebase Admin inizializzato con successo.");
  } catch (error) {
    console.error("Errore critico durante l'inizializzazione di Firebase:", error);
  }
}

// ==============================================================================
// GESTORE DELLA RICHIESTA HTTP (ENDPOINT VERCEL)
// ==============================================================================
// Questa è la funzione principale che Vercel esegue quando Flutter fa la chiamata http.post
module.exports = async (req, res) => {
  // Sicurezza: accettiamo solo richieste POST (che contengono dati).
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Metodo non consentito. Usa POST.' });
  }

  try {
    // Estrae i dati inviati da Flutter nel payload JSON
    const { mittente, link_maps } = req.body;

    if (!mittente) {
      return res.status(400).json({ error: 'ID mittente mancante nella richiesta.' });
    }

    const db = admin.firestore();

    // ==============================================================================
    // 1. RECUPERO DEI CONTATTI DI EMERGENZA
    // ==============================================================================
    // Cerca nel database la sottocartella dei contatti per questo specifico mittente.
    const contattiSnapshot = await db
      .collection('utenti')
      .doc(mittente)
      .collection('contatti_emergenza')
      .get();

    if (contattiSnapshot.empty) {
      // Ritorna codice 404 (Not Found) se l'utente non ha impostato nessun contatto.
      return res.status(404).json({ message: 'Nessun contatto di emergenza trovato per questo utente.' });
    }

    // ==============================================================================
    // 2. RECUPERO DEI TOKEN FCM (INDIRIZZI DEI TELEFONI)
    // ==============================================================================
    const fcmTokens = [];
    
    // Cicla attraverso ogni contatto di emergenza trovato
    for (const doc of contattiSnapshot.docs) {
      const idContatto = doc.id; // L'ID utente del contatto
      
      // Va a leggere il documento principale del contatto per trovare il suo fcmToken
      const utenteContatto = await db.collection('utenti').doc(idContatto).get();
      const token = utenteContatto.data()?.fcmToken;
      
      // Se il contatto ha un token salvato (ha installato e aperto l'app), lo aggiunge alla lista.
      if (token) {
        fcmTokens.push(token);
      }
    }

    if (fcmTokens.length === 0) {
      return res.status(404).json({ message: 'Nessun dispositivo valido (Token FCM) trovato tra i contatti.' });
    }

    // ==============================================================================
    // 3. COSTRUZIONE E INVIO DELLA NOTIFICA PUSH
    // ==============================================================================
    const payload = {
      notification: {
        title: "🚨 EMERGENZA EVA!",
        body: "Richiesta di aiuto. Tocca per aprire la posizione."
      },
      data: {
        // Dati invisibili utili se l'app vuole elaborare il link internamente
        linkMaps: link_maps || "", 
        mittente: mittente,
        timestamp: new Date().toISOString()
      }
    };

    // 'sendEachForMulticast' invia la stessa notifica contemporaneamente a tutta la lista di token
    const response = await admin.messaging().sendEachForMulticast({
      tokens: fcmTokens,
      notification: payload.notification,
      data: payload.data,
    });

    // Se tutto va a buon fine, risponde a Flutter con codice 200 (Successo)
    return res.status(200).json({ 
      success: true, 
      message: 'Notifiche inviate con successo',
      notificheInviate: response.successCount,
      notificheFallite: response.failureCount
    });

  } catch (error) {
    console.error("Errore interno del server durante l'elaborazione dell'allarme:", error);
    // In caso di crash del codice o disconnessione dal database, restituisce codice 500
    return res.status(500).json({ error: "Errore fatale del server." });
  }
};
*/