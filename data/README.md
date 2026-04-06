# data/

## Expected files

| File | Description |
|---|---|
| `example_mixture.wav` | Short 4-channel example recording (for quick tests) |
| `mixture.wav` | Full 4-channel hearing-aid recording (main input) |

Both files must be **4-channel, 44 100 Hz, WAV** format.

Channel order (index → label):

| Index | Label | Position |
|---|---|---|
| 0 | LF | Left-Front |
| 1 | LR | Left-Rear |
| 2 | RF | Right-Front |
| 3 | RR | Right-Rear |

## Important

- **Do NOT commit** WAV files to Git — they are listed in `.gitignore`.
- Place them manually in this directory, or download them from the shared
  drive / LMS / whatever the team uses.
- If you need to share audio for testing, use the project's shared cloud
  storage and document the link here.
