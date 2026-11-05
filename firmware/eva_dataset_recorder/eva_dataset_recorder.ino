// XIAO ESP32-S3 Sense: registra WAV mono PCM16 16kHz su microSD.
// Invia il nome completo del WAV, poi rec per registrare subito 10 secondi.
// Nessun prefisso, suffisso o numero progressivo aggiunto al nome (salvo .wav se manca).
#include <Arduino.h>
#include <ESP_I2S.h>
#include <FS.h>
#include <SD.h>
#include <SPI.h>
#include <esp_heap_caps.h>

constexpr uint32_t sample_rate=16000,record_seconds=10;
constexpr size_t samples_count=sample_rate*record_seconds;
constexpr size_t audio_bytes=samples_count*sizeof(int16_t);
constexpr int volume_gain=2;
static I2SClass microphone;
static int16_t* audio=nullptr;
static String pending_name="",command="";
static bool command_overflow=false;
static bool ready=false;

void help() {
  Serial.println("\nEVA DATASET RECORDER v2 - WAV 10s, mono PCM16 16000Hz, gain2");
  Serial.println("1. Invia il nome completo, per esempio unknown_P15_S01_R01_solo-eva_esp.wav");
  Serial.println("2. Invia rec: registra 10 secondi, senza conto alla rovescia.");
  Serial.println("3. Parla quando compare REGISTRAZIONE IN CORSO e attendi SALVATO.");
  Serial.println("File in /recordings. .wav aggiunto solo se manca. Nessuna sovrascrittura.");
  Serial.println("h = istruzioni. Monitor seriale: 115200 baud, New Line.");
  Serial.printf("Nome impostato=%s; SD=%s\n",pending_name.length()?pending_name.c_str():"NESSUNO",ready?"OK":"NON PRONTA");
}
bool valid_filename(const String &name) {
  if(name.length()<5 || name.length()>120 || name[0]=='.') return false;
  for(unsigned int i=0;i<name.length();++i) {
    char c=name[i];
    if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.')) return false;
  }
  return name.indexOf("..")<0;
}
void le16(uint8_t* p,uint16_t value) {p[0]=value;p[1]=value>>8;}
void le32(uint8_t* p,uint32_t value) {for(int i=0;i<4;++i)p[i]=value>>(8*i);}
void wav_header(uint8_t* h) {
  memset(h,0,44);memcpy(h,"RIFF",4);le32(h+4,audio_bytes+36);
  memcpy(h+8,"WAVEfmt ",8);le32(h+16,16);le16(h+20,1);le16(h+22,1);
  le32(h+24,sample_rate);le32(h+28,sample_rate*2);le16(h+32,2);le16(h+34,16);
  memcpy(h+36,"data",4);le32(h+40,audio_bytes);
}
void record() {
  if(!ready) {Serial.println("SD/buffer non pronti. Correggere e riavviare.");return;}
  if(!pending_name.length()) {Serial.println("Prima invia il nome completo del file, poi rec.");return;}
  String filename="/recordings/"+pending_name;
  if(SD.exists(filename)){Serial.println("ERRORE: nome gia presente. Invia un nuovo nome.");pending_name="";return;}
  Serial.printf("\nProssimo file: %s\n",filename.c_str());
  // Avvio immediato: solo inizializzazione e assestamento del microfono prima del segnale.
  microphone.setPinsPdmRx(42,41);
  if(!microphone.begin(I2S_MODE_PDM_RX,sample_rate,I2S_DATA_BIT_WIDTH_16BIT,I2S_SLOT_MODE_MONO)) {
    Serial.println("ERRORE microfono PDM.");return;
  }
  int16_t warmup[512];
  for(int i=0;i<4;++i) {
    if(microphone.readBytes((char*)warmup,sizeof(warmup))!=sizeof(warmup)) {
      microphone.end();Serial.println("ERRORE avvio microfono. Nessun file salvato.");return;
    }
  }
  Serial.println("REGISTRAZIONE IN CORSO - 10 secondi");
  size_t total=0;uint32_t start=millis();
  while(total<audio_bytes && millis()-start<(record_seconds+3)*1000) {
    size_t length=min((size_t)1024,audio_bytes-total);
    size_t got=microphone.readBytes((char*)audio+total,length);
    if(got!=length) break;
    total+=got;
  }
  microphone.end();
  if(total!=audio_bytes) {Serial.printf("ERRORE lettura: %u/%u byte. Nessun WAV salvato.\n",(unsigned)total,(unsigned)audio_bytes);return;}
  Serial.println("Registrazione terminata. Salvataggio sulla SD...");
  unsigned int clipped=0;double energy=0;
  for(size_t i=0;i<samples_count;++i) {
    int value=(int)audio[i]*volume_gain;
    if(value>32767){value=32767;++clipped;}
    if(value< -32768){value=-32768;++clipped;}
    audio[i]=(int16_t)value;energy+=(double)value*value;
  }
  // Controllo aggiuntivo: FILE_WRITE non deve sovrascrivere file esistenti.
  if(SD.exists(filename)){Serial.println("ERRORE: il file esiste gia. Nessuna scrittura.");return;}
  File file=SD.open(filename,FILE_WRITE);
  if(!file){Serial.println("ERRORE apertura file SD.");return;}
  uint8_t header[44];wav_header(header);
  bool ok=file.write(header,44)==44;
  if(ok) ok=file.write((const uint8_t*)audio,audio_bytes)==audio_bytes;
  file.flush();size_t size=file.size();file.close();
  if(!ok || size!=audio_bytes+44) {
    Serial.println("ERRORE scrittura incompleta: rimosso solo il nuovo file incompleto.");SD.remove(filename);return;
  }
  Serial.printf("SALVATO: %s (%u byte) RMS=%.1f, saturati=%u\n",filename.c_str(),(unsigned)size,sqrt(energy/samples_count),clipped);
  pending_name="";
  Serial.println("Invia il nome del prossimo file, poi rec. Attendi SALVATO prima di scollegare.");
}
void execute(String value) {
  value.trim();
  String lower=value;lower.toLowerCase();
  if(lower=="rec"){record();return;}
  if(lower=="h" || lower=="help"){help();return;}
  // Qualsiasi altra riga e il nome desiderato: non inferire label o identita.
  pending_name="";
  if(!lower.endsWith(".wav")) value+=".wav";
  if(!valid_filename(value)) {
    Serial.println("Nome non valido: usa lettere, numeri, _ - e punti, senza cartelle o spazi (max 120 caratteri).");return;
  }
  if(!ready){Serial.println("SD/buffer non pronti. Correggere e riavviare.");return;}
  if(SD.exists("/recordings/"+value)) {
    Serial.println("Il file esiste gia: nessuna sovrascrittura. Invia un altro nome.");return;
  }
  pending_name=value;
  Serial.printf("NOME IMPOSTATO: %s\nInvia rec per registrare 10 secondi.\n",pending_name.c_str());
}
void setup() {
  Serial.begin(115200);uint32_t start=millis();while(!Serial && millis()-start<15000)delay(10);
  if(!SD.begin(21)) {Serial.println("ERRORE SD: inserire microSD FAT32 e riavviare.");help();return;}
  if(!SD.exists("/recordings") && !SD.mkdir("/recordings")) {Serial.println("ERRORE creazione cartella.");help();return;}
  audio=(int16_t*)heap_caps_malloc(audio_bytes,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
  if(!audio) {Serial.println("ERRORE buffer: attivare Tools > PSRAM > OPI PSRAM e ricaricare.");help();return;}
  ready=true;help();
}
void loop() {
  while(Serial.available()) {
    char c=Serial.read();
    if(c=='\n'||c=='\r') {
      if(command_overflow){pending_name="";Serial.println("Riga troppo lunga: nome annullato, invia di nuovo.");}
      else if(command.length()) execute(command);
      command="";command_overflow=false;
    } else if(!command_overflow) {
      if(command.length()<120) command+=c;
      else {command_overflow=true;command="";}
    }
  }
  delay(10);
}
