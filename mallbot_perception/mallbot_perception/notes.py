"""
NoteBook: turns many noisy per-frame detections into one note per real object.

  observation (label, x, y, z in map)  ->  match an existing note of the same label within a
  distance gate  ->  update its running-average position and sighting count, or start a new note.

  - a note is CONFIRMED only after `min_sightings` observations (one-frame false positives never
    become notes)
  - DYNAMIC classes (person, ...) are never written as notes; the caller just shows them
  - merge() joins notes of the same label that drifted together (duplicates)
Pure Python (no ROS) so it can be unit-tested offline.
"""
import json
import math
import time


class Note:
    __slots__ = ("id", "label", "x", "y", "z", "n", "conf", "first_seen", "last_seen", "text")

    def __init__(self, nid, label, x, y, z, conf, t):
        self.id, self.label = nid, label
        self.x, self.y, self.z = x, y, z
        self.n, self.conf = 1, conf
        self.first_seen = self.last_seen = t
        self.text = ""            # filled later by OCR for signs

    def add(self, x, y, z, conf, t):
        self.n += 1
        a = 1.0 / self.n          # running mean (every sighting counts equally)
        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        self.z += a * (z - self.z)
        self.conf = max(self.conf, conf)
        self.last_seen = t

    def dist2d(self, x, y):
        return math.hypot(self.x - x, self.y - y)

    def to_dict(self):
        return {"id": self.id, "label": self.label, "x": round(self.x, 3), "y": round(self.y, 3),
                "z": round(self.z, 3), "sightings": self.n, "conf": round(self.conf, 3),
                "first_seen": round(self.first_seen, 2), "last_seen": round(self.last_seen, 2),
                "text": self.text}


class NoteBook:
    def __init__(self, gate=0.8, class_gates=None, min_sightings=3, dynamic=("person",)):
        self.gate = gate
        self.class_gates = dict(class_gates or {})
        self.min_sightings = min_sightings
        self.dynamic = set(dynamic)
        self.notes = []
        self._next = 1

    def gate_for(self, label):
        return self.class_gates.get(label, self.gate)

    def is_dynamic(self, label):
        return label in self.dynamic

    def observe(self, label, x, y, z, conf=1.0, t=None):
        """Add one detection in the map frame. Returns the Note, or None for dynamic classes."""
        if self.is_dynamic(label):
            return None
        t = time.time() if t is None else t
        g = self.gate_for(label)
        best, best_d = None, g
        for nt in self.notes:
            if nt.label == label:
                d = nt.dist2d(x, y)
                if d < best_d:
                    best, best_d = nt, d
        if best is None:
            best = Note(self._next, label, x, y, z, conf, t)
            self._next += 1
            self.notes.append(best)
        else:
            best.add(x, y, z, conf, t)
        return best

    def merge(self):
        """Join same-label notes closer than 0.75 x gate (sighting-weighted). Returns merges done."""
        merged = 0
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(self.notes):
                for b in self.notes[i + 1:]:
                    if a.label == b.label and a.dist2d(b.x, b.y) < 0.75 * self.gate_for(a.label):
                        w = a.n + b.n
                        a.x = (a.x * a.n + b.x * b.n) / w
                        a.y = (a.y * a.n + b.y * b.n) / w
                        a.z = (a.z * a.n + b.z * b.n) / w
                        a.n = w
                        a.conf = max(a.conf, b.conf)
                        a.first_seen = min(a.first_seen, b.first_seen)
                        a.last_seen = max(a.last_seen, b.last_seen)
                        a.text = a.text or b.text
                        self.notes.remove(b)
                        merged += 1
                        changed = True
                        break
                if changed:
                    break
        return merged

    def confirmed(self):
        return [n for n in self.notes if n.n >= self.min_sightings]

    def to_json(self, path, meta=None):
        data = {"meta": meta or {}, "notes": [n.to_dict() for n in self.confirmed()]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        import os
        os.replace(tmp, path)          # atomic: never leaves a half-written file
        return len(data["notes"])
