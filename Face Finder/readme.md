Given some baseline photos of a single person the task is to retrieve each photo with that person.

Firstly we need to run save_baseline.ipynb which save the embedding of the faces from baseline photo.

After we run find_faces.ipynb which rescale (if needed) and search for specific faces which are similar to the baseline one.

We select each photos which has a face with at least 50% of match from baseline faces.

With save_output.ipynb we save the selected photos.
