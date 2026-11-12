# Sample media ("virtual cameras")
Put demo videos / images here (mp4, avi, mov, jpg, png). They appear in the
"Sample media" drop-down on the Upload page and can be processed by
`python scripts/batch_process.py`.

Where to get footage (no live camera needed):
1. Self-recorded staged scenarios (recommended for validation - you know the ground truth):
   2-4 classmates, helmet/vest on/off, walking into a taped "restricted zone", near a parked vehicle.
   Record 30-60 s clips with a phone at 1080p, camera fixed on a tripod at 3-5 m height if possible.
2. Test-split images of the training dataset (Roboflow Construction Site Safety).
3. Public construction / mining clips with a reuse-permitting licence (e.g. Creative Commons on
   Pexels / Pixabay / Wikimedia Commons). Record the source URL and licence in sources.csv.
Keep each demo clip short (20-60 s) so CPU processing finishes in 1-3 minutes.
