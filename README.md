# Sign Language Detector

A browser-based real-time detector trained for the letters `A`, `B`, `C`, `D`,
`E`, `H`, and `R`.

## Setup on Windows

Use Python 3.10. The old `myvenv` folder was created on a different computer
and should not be used.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Double-click `run_app.bat` to run the reliable native desktop camera app.
Press `C` to clear the sentence and `Q` or `Esc` to quit.

The old browser launcher now redirects to the desktop application because the
WebRTC transport was not reliable on this machine.

## Application modes

### Detection

- Recognizes trained signs in real time.
- Shows prediction confidence and builds a detected sequence.
- Use `Ctrl+L` to clear the sequence.

### Training

1. Open the `Training` tab.
2. Select a letter from `A` to `Z`.
3. Click `Start capture`.
4. Hold the sign inside the guide and slowly vary angle and distance.
5. After 60 samples, the classifier rebuilds automatically.
6. Return to `Detection`; the trained sign is available immediately.

Training stores MediaPipe hand landmarks in `MP_Data`. It does not save camera
photos or video.

Open the displayed local URL, allow camera access, and hold one hand inside the
green box. The current model only recognizes the seven labels listed above.

## Training data

`MP_Data` contains 30 sequences of 30 hand-landmark frames for each trained
label. Run `trainmodel.py` only when you intentionally want to retrain and
replace `model.json` and `model.h5`.
