"""One-off: построить боевой coarse-энкодер + FAISS-индекс по текущей базе."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workers.tasks.index_task import build_vocabulary, build_index

print("building vocabulary (dino_vlad AnyLoc k-means)...", flush=True)
v = build_vocabulary.apply().get()
print("vocab:", v, flush=True)

print("building FAISS index...", flush=True)
i = build_index.apply().get()
print("index:", i, flush=True)
print("DONE", flush=True)
