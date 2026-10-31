# Edge Impulse MFCC source

Repository: https://github.com/edgeimpulse/processing-blocks
Commit: 9b0ceb2b5d22658d3193a5b77dc773cfc8d4cab0
Files: mfcc/third_party/speechpy/*.py, preserving header notices.
Reference pipeline: mfcc/dsp.py at the same commit.
Local compatibility patches: np.lib.pad -> np.pad (NumPy 2); remove sys.path mutation;
replace import common.errors by local ConfigurationError(ValueError).
Numerical algorithm unchanged. Version 4 is the assumed block version, not confirmed
against this user's project export.
