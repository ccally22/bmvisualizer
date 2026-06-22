"""
anny_wrapper.py

Wrapper for the ANNY body model.
Makes ANNY usable in the Body Model Visualizer next to SMPL, SUPR, STAR, etc.

ANNY has a different API:
- uses bone matrices (4x4) instead of axis-angle tensors
- uses phenotype params (age, height, ...) instead of betas
- uses Z-up coordinate system instead of Y-up

This wrapper translates between the two so main.py doesn't need to care.
"""

#imports
import torch
import numpy as np
import anny
from scipy.spatial.transform import Rotation as R

class AnnyOutput:
    """
    Fake SMPL-style output container.
    Has .vertices and .joints so main.py can use it like SMPL output.
    """
    def __init__(self, vertices, joints):
        self.vertices = vertices
        self.joints = joints

class AnnyWrapper:
    """
    Wraps the ANNY body model with a SMPL-style API.
    
    Lets main.py call ANNY just like SMPL/SUPR/STAR:
        model(betas=..., pose=..., trans=...)
    
    Translates between:
    - SMPL-style pose tensor [N*3] <-> ANNY bone matrix dict
    - ANNY bone_poses [4x4] <-> SMPL-style joints [3]
    - ANNY Z-up <-> visualizer Y-up
    """
    
    def __init__(self, gender='neutral', rig='default'):
        # Load the ANNY model itself
        self.anny_model = anny.create_fullbody_model(
            rig=rig,
            local_changes=True,
            all_phenotypes=True,
            triangulate_faces=True,
        )
        
        # Remember which gender/rig we use
        self.gender = gender
        self.rig = rig
        
        # SMPL-compatible attributes (main.py uses these)
        # faces need to be numpy for open3d
        self.faces = self.anny_model.faces.detach().numpy()
        
        # Bone info (for GUI later)
        self.bone_labels = self.anny_model.bone_labels
        self.n_bones = len(self.bone_labels)
        
        # Storage for ANNY-specific features (set via setters later)
        self.phenotypes = {}
        self.local_changes = {}



