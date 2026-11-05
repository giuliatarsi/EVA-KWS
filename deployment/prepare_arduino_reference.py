# Aprire in Spyder ed eseguire tutto il file con F5.
"""Genera un riferimento WAV -> MFCC -> CNN per il porting Arduino.
Non esegue training e non modifica il modello.
"""
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
# None: usa il training finale completo più recente.
RUN_FOLDER = None
# None: seleziona automaticamente un campione EVA della validation.
WAV_PATH = None


def main():
    import csv,json,numpy as np
    from training.config import MFCCConfig,write_json
    from training.features import read_clip,mfcc
    from deployment.tflite_export import lite_predictions
    root=PROJECT/'results'/'final_model'
    run=Path(RUN_FOLDER) if RUN_FOLDER else max(
        (p for p in root.iterdir() if p.is_dir() and (p/'complete.json').exists()),
        key=lambda p:p.name)
    deployment=json.loads((run/'deployment.json').read_text())
    records=list(csv.DictReader((run/'manifest.csv').open()))
    if WAV_PATH:
        path=Path(WAV_PATH).resolve()
        record=next((r for r in records if Path(r['path']).resolve()==path),None)
        if record is None:
            record=dict(path=str(path),relative=path.name)
    else:
        record=next(r for r in records if r['subset']=='validation' and r['label']=='eva')
    cfg=MFCCConfig(**deployment['mfcc'])
    audio=read_clip(record,cfg)
    features=mfcc(audio,cfg)
    reference={}
    for mode in ['float32','int8']:
        scores,io=lite_predictions(run/f'eva_{mode}.tflite',features[None])
        reference[mode]=dict(scores=scores[0].tolist(),prediction=deployment['labels'][int(scores.argmax())],io=io)
    out=run/'arduino_reference';out.mkdir(exist_ok=True)
    np.save(out/'reference_mfcc.npy',features)
    write_json(out/'reference_expected.json',dict(wav=record['path'],labels=deployment['labels'],
        mfcc_shape=list(features.shape),mfcc=deployment['mfcc'],predictions=reference))
    import shutil
    shutil.copyfile(record['path'],out/'reference.wav')
    def array(name,ctype,values,fmt):
        items=[fmt(v) for v in values]
        lines=[', '.join(items[i:i+12]) for i in range(0,len(items),12)]
        return f'alignas(16) const {ctype} {name}[] = {{\n    '+',\n    '.join(lines)+'\n};\n'
    header='#pragma once\n#include <stdint.h>\n#include <stddef.h>\n'
    header+='extern const int16_t reference_audio[];\nextern const float reference_mfcc[];\n'
    header+=f'constexpr size_t reference_audio_count = {len(audio)};\nconstexpr size_t reference_mfcc_count = {features.size};\n'
    (out/'reference_data.h').write_text(header)
    source='#include "reference_data.h"\n'
    source+=array('reference_audio','int16_t',audio,lambda v:str(int(v)))
    source+=array('reference_mfcc','float',features.ravel(),lambda v:f'{float(v):.9e}f')
    (out/'reference_data.cpp').write_text(source)
    for mode in ['float32','int8']:
        name=f'eva_{mode}_model'
        data=(run/f'eva_{mode}.tflite').read_bytes()
        (out/f'{name}.h').write_text('#pragma once\n#include <stddef.h>\nextern const unsigned char '+name+'[];\n'+f'constexpr size_t {name}_length = {len(data)};\n')
        (out/f'{name}.cpp').write_text(f'#include "{name}.h"\n'+array(name,'unsigned char',data,lambda v:f'0x{v:02x}'))
    print('\nRiferimento pronto:',out)
    print('Audio:',record['path'])
    print('MFCC:',features.shape)
    for mode, result in reference.items():print(mode,result['prediction'],dict(zip(deployment['labels'],result['scores'])))
    print('Questi file sono dati per il futuro sketch, NON ancora un programma Arduino completo.')
    return out

if __name__=='__main__':
    main()
