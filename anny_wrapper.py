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

    def _z_up_to_y_up(self, points):
        """
        Convert Z-up coordinates (ANNY/Blender) to Y-up (SMPL/visualizer).
        Rotation by 90 degrees around the X-axis.
        
        Works for any shape ending in 3: [3], [N, 3], [B, N, 3], ...
        """
        result = points.clone()
        result[..., 1] = points[..., 2]    # new Y = old Z
        result[..., 2] = -points[..., 1]   # new Z = -old Y
        return result

    def _axis_angle_to_matrix(self, axis_angle):
        """
        Convert an axis-angle vector to a 3x3 rotation matrix (Rodrigues formula)
        
        Input:  tensor [3] (axis-angle: direction = axis, magnitude = angle in radians)
        Output: tensor [3, 3] (rotation matrix)
        """
        # Use scipy to do the Rodrigues conversion
        axis_angle_np = axis_angle.detach().numpy()
        rot_matrix = R.from_rotvec(axis_angle_np).as_matrix()
        return torch.from_numpy(rot_matrix).float()

    def set_phenotypes(self, phenotypes_dict):
        """
        Set ANNY phenotype parameters from outside.
        Called by main.py to push GUI values to the wrapper.
        """
        self.phenotypes = phenotypes_dict
    
    def set_local_changes(self, local_changes_dict):
        """
        Set ANNY local change parameters from outside.
        Called by main.py to push GUI values to the wrapper.
        """
        self.local_changes = local_changes_dict
    
    def set_rig(self, rig_name):
        """
        Change the rig (default, mixamo, cmu_mb, etc.).
        Reloads the ANNY model with the new rig.
        """
        if rig_name != self.rig:
            self.rig = rig_name
            self.anny_model = anny.create_fullbody_model(
                rig=rig_name,
                local_changes=True,
                all_phenotypes=True,
                triangulate_faces=True,
            )
            # Update bone info because rigs can have different bones!
            self.bone_labels = self.anny_model.bone_labels
            self.n_bones = len(self.bone_labels)
            self.faces = self.anny_model.faces.detach().numpy()


