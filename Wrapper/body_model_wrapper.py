from abc import ABC, abstractmethod

class BodyModelWrapper(ABC):
    @abstractmethod
    def preload_body_model(self, gender):
        # gibt das model zurück
        pass

    @abstractmethod
    def forward(self, input_params, shape_params=None, expression=None):
        # gibt verts, joints, faces zurück als tupel mesh_data
        # type numpy.ndarray
        # shape_params: betas (Tensor) für SMPL-Familie, phenotypes (Dict) für ANNY
        pass