from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PredictRequest(_message.Message):
    __slots__ = ("smiles", "molecule_id", "return_features", "return_interpretation", "options")
    class OptionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SMILES_FIELD_NUMBER: _ClassVar[int]
    MOLECULE_ID_FIELD_NUMBER: _ClassVar[int]
    RETURN_FEATURES_FIELD_NUMBER: _ClassVar[int]
    RETURN_INTERPRETATION_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    molecule_id: str
    return_features: bool
    return_interpretation: bool
    options: _containers.ScalarMap[str, str]
    def __init__(self, smiles: _Optional[str] = ..., molecule_id: _Optional[str] = ..., return_features: bool = ..., return_interpretation: bool = ..., options: _Optional[_Mapping[str, str]] = ...) -> None: ...

class PBPKPrediction(_message.Message):
    __slots__ = ("fu", "vd", "clearance", "cl_hepatic", "cl_renal", "cl_intrinsic", "fu_std", "vd_std", "clearance_std", "confidence", "applicability_domain")
    FU_FIELD_NUMBER: _ClassVar[int]
    VD_FIELD_NUMBER: _ClassVar[int]
    CLEARANCE_FIELD_NUMBER: _ClassVar[int]
    CL_HEPATIC_FIELD_NUMBER: _ClassVar[int]
    CL_RENAL_FIELD_NUMBER: _ClassVar[int]
    CL_INTRINSIC_FIELD_NUMBER: _ClassVar[int]
    FU_STD_FIELD_NUMBER: _ClassVar[int]
    VD_STD_FIELD_NUMBER: _ClassVar[int]
    CLEARANCE_STD_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    APPLICABILITY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    fu: float
    vd: float
    clearance: float
    cl_hepatic: float
    cl_renal: float
    cl_intrinsic: float
    fu_std: float
    vd_std: float
    clearance_std: float
    confidence: float
    applicability_domain: float
    def __init__(self, fu: _Optional[float] = ..., vd: _Optional[float] = ..., clearance: _Optional[float] = ..., cl_hepatic: _Optional[float] = ..., cl_renal: _Optional[float] = ..., cl_intrinsic: _Optional[float] = ..., fu_std: _Optional[float] = ..., vd_std: _Optional[float] = ..., clearance_std: _Optional[float] = ..., confidence: _Optional[float] = ..., applicability_domain: _Optional[float] = ...) -> None: ...

class PBPKPredictionResponse(_message.Message):
    __slots__ = ("success", "prediction", "quantum_features", "docking_scores", "transporter_preds", "interpretation", "error", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_FIELD_NUMBER: _ClassVar[int]
    QUANTUM_FEATURES_FIELD_NUMBER: _ClassVar[int]
    DOCKING_SCORES_FIELD_NUMBER: _ClassVar[int]
    TRANSPORTER_PREDS_FIELD_NUMBER: _ClassVar[int]
    INTERPRETATION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    success: bool
    prediction: PBPKPrediction
    quantum_features: QuantumFeatures
    docking_scores: DockingScores
    transporter_preds: TransporterPredictions
    interpretation: Interpretation
    error: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, success: bool = ..., prediction: _Optional[_Union[PBPKPrediction, _Mapping]] = ..., quantum_features: _Optional[_Union[QuantumFeatures, _Mapping]] = ..., docking_scores: _Optional[_Union[DockingScores, _Mapping]] = ..., transporter_preds: _Optional[_Union[TransporterPredictions, _Mapping]] = ..., interpretation: _Optional[_Union[Interpretation, _Mapping]] = ..., error: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class QuantumRequest(_message.Message):
    __slots__ = ("smiles", "basis_set", "method")
    SMILES_FIELD_NUMBER: _ClassVar[int]
    BASIS_SET_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    basis_set: str
    method: str
    def __init__(self, smiles: _Optional[str] = ..., basis_set: _Optional[str] = ..., method: _Optional[str] = ...) -> None: ...

class QuantumFeatures(_message.Message):
    __slots__ = ("homo_energy", "lumo_energy", "homo_lumo_gap", "chemical_hardness", "chemical_softness", "electronegativity", "electrophilicity_index", "fukui_electrophilic_max", "fukui_nucleophilic_max", "fukui_radical_max", "fukui_electrophilic_mean", "dipole_moment", "dipole_x", "dipole_y", "dipole_z", "ionization_potential", "electron_affinity", "polarizability_iso", "polarizability_aniso", "basis_set_used", "scf_cycles", "computation_time", "converged")
    HOMO_ENERGY_FIELD_NUMBER: _ClassVar[int]
    LUMO_ENERGY_FIELD_NUMBER: _ClassVar[int]
    HOMO_LUMO_GAP_FIELD_NUMBER: _ClassVar[int]
    CHEMICAL_HARDNESS_FIELD_NUMBER: _ClassVar[int]
    CHEMICAL_SOFTNESS_FIELD_NUMBER: _ClassVar[int]
    ELECTRONEGATIVITY_FIELD_NUMBER: _ClassVar[int]
    ELECTROPHILICITY_INDEX_FIELD_NUMBER: _ClassVar[int]
    FUKUI_ELECTROPHILIC_MAX_FIELD_NUMBER: _ClassVar[int]
    FUKUI_NUCLEOPHILIC_MAX_FIELD_NUMBER: _ClassVar[int]
    FUKUI_RADICAL_MAX_FIELD_NUMBER: _ClassVar[int]
    FUKUI_ELECTROPHILIC_MEAN_FIELD_NUMBER: _ClassVar[int]
    DIPOLE_MOMENT_FIELD_NUMBER: _ClassVar[int]
    DIPOLE_X_FIELD_NUMBER: _ClassVar[int]
    DIPOLE_Y_FIELD_NUMBER: _ClassVar[int]
    DIPOLE_Z_FIELD_NUMBER: _ClassVar[int]
    IONIZATION_POTENTIAL_FIELD_NUMBER: _ClassVar[int]
    ELECTRON_AFFINITY_FIELD_NUMBER: _ClassVar[int]
    POLARIZABILITY_ISO_FIELD_NUMBER: _ClassVar[int]
    POLARIZABILITY_ANISO_FIELD_NUMBER: _ClassVar[int]
    BASIS_SET_USED_FIELD_NUMBER: _ClassVar[int]
    SCF_CYCLES_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_TIME_FIELD_NUMBER: _ClassVar[int]
    CONVERGED_FIELD_NUMBER: _ClassVar[int]
    homo_energy: float
    lumo_energy: float
    homo_lumo_gap: float
    chemical_hardness: float
    chemical_softness: float
    electronegativity: float
    electrophilicity_index: float
    fukui_electrophilic_max: float
    fukui_nucleophilic_max: float
    fukui_radical_max: float
    fukui_electrophilic_mean: float
    dipole_moment: float
    dipole_x: float
    dipole_y: float
    dipole_z: float
    ionization_potential: float
    electron_affinity: float
    polarizability_iso: float
    polarizability_aniso: float
    basis_set_used: str
    scf_cycles: int
    computation_time: float
    converged: bool
    def __init__(self, homo_energy: _Optional[float] = ..., lumo_energy: _Optional[float] = ..., homo_lumo_gap: _Optional[float] = ..., chemical_hardness: _Optional[float] = ..., chemical_softness: _Optional[float] = ..., electronegativity: _Optional[float] = ..., electrophilicity_index: _Optional[float] = ..., fukui_electrophilic_max: _Optional[float] = ..., fukui_nucleophilic_max: _Optional[float] = ..., fukui_radical_max: _Optional[float] = ..., fukui_electrophilic_mean: _Optional[float] = ..., dipole_moment: _Optional[float] = ..., dipole_x: _Optional[float] = ..., dipole_y: _Optional[float] = ..., dipole_z: _Optional[float] = ..., ionization_potential: _Optional[float] = ..., electron_affinity: _Optional[float] = ..., polarizability_iso: _Optional[float] = ..., polarizability_aniso: _Optional[float] = ..., basis_set_used: _Optional[str] = ..., scf_cycles: _Optional[int] = ..., computation_time: _Optional[float] = ..., converged: bool = ...) -> None: ...

class QuantumFeaturesResponse(_message.Message):
    __slots__ = ("success", "features", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    features: QuantumFeatures
    error: str
    def __init__(self, success: bool = ..., features: _Optional[_Union[QuantumFeatures, _Mapping]] = ..., error: _Optional[str] = ...) -> None: ...

class DockingRequest(_message.Message):
    __slots__ = ("smiles", "isoforms", "n_runs")
    SMILES_FIELD_NUMBER: _ClassVar[int]
    ISOFORMS_FIELD_NUMBER: _ClassVar[int]
    N_RUNS_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    isoforms: _containers.RepeatedScalarFieldContainer[str]
    n_runs: int
    def __init__(self, smiles: _Optional[str] = ..., isoforms: _Optional[_Iterable[str]] = ..., n_runs: _Optional[int] = ...) -> None: ...

class CYPDockingScore(_message.Message):
    __slots__ = ("isoform", "binding_energy", "distance_to_heme", "h_bonds", "hydrophobic_contacts", "binding_mode", "confidence")
    ISOFORM_FIELD_NUMBER: _ClassVar[int]
    BINDING_ENERGY_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_TO_HEME_FIELD_NUMBER: _ClassVar[int]
    H_BONDS_FIELD_NUMBER: _ClassVar[int]
    HYDROPHOBIC_CONTACTS_FIELD_NUMBER: _ClassVar[int]
    BINDING_MODE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    isoform: str
    binding_energy: float
    distance_to_heme: float
    h_bonds: int
    hydrophobic_contacts: int
    binding_mode: str
    confidence: float
    def __init__(self, isoform: _Optional[str] = ..., binding_energy: _Optional[float] = ..., distance_to_heme: _Optional[float] = ..., h_bonds: _Optional[int] = ..., hydrophobic_contacts: _Optional[int] = ..., binding_mode: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class DockingScores(_message.Message):
    __slots__ = ("scores", "best_binding_energy", "best_isoform", "promiscuity_score", "weighted_score")
    SCORES_FIELD_NUMBER: _ClassVar[int]
    BEST_BINDING_ENERGY_FIELD_NUMBER: _ClassVar[int]
    BEST_ISOFORM_FIELD_NUMBER: _ClassVar[int]
    PROMISCUITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    WEIGHTED_SCORE_FIELD_NUMBER: _ClassVar[int]
    scores: _containers.RepeatedCompositeFieldContainer[CYPDockingScore]
    best_binding_energy: float
    best_isoform: str
    promiscuity_score: float
    weighted_score: float
    def __init__(self, scores: _Optional[_Iterable[_Union[CYPDockingScore, _Mapping]]] = ..., best_binding_energy: _Optional[float] = ..., best_isoform: _Optional[str] = ..., promiscuity_score: _Optional[float] = ..., weighted_score: _Optional[float] = ...) -> None: ...

class DockingScoresResponse(_message.Message):
    __slots__ = ("success", "scores", "error", "computation_time")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_TIME_FIELD_NUMBER: _ClassVar[int]
    success: bool
    scores: DockingScores
    error: str
    computation_time: float
    def __init__(self, success: bool = ..., scores: _Optional[_Union[DockingScores, _Mapping]] = ..., error: _Optional[str] = ..., computation_time: _Optional[float] = ...) -> None: ...

class TransporterRequest(_message.Message):
    __slots__ = ("smiles", "transporters")
    SMILES_FIELD_NUMBER: _ClassVar[int]
    TRANSPORTERS_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    transporters: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, smiles: _Optional[str] = ..., transporters: _Optional[_Iterable[str]] = ...) -> None: ...

class TransporterPredictions(_message.Message):
    __slots__ = ("pgp_efflux_prob", "bcrp_efflux_prob", "mrp2_efflux_prob", "oatp1b1_uptake_prob", "oatp1b3_uptake_prob", "oat1_uptake_prob", "oat3_uptake_prob", "oct2_uptake_prob", "efflux_score", "uptake_score")
    PGP_EFFLUX_PROB_FIELD_NUMBER: _ClassVar[int]
    BCRP_EFFLUX_PROB_FIELD_NUMBER: _ClassVar[int]
    MRP2_EFFLUX_PROB_FIELD_NUMBER: _ClassVar[int]
    OATP1B1_UPTAKE_PROB_FIELD_NUMBER: _ClassVar[int]
    OATP1B3_UPTAKE_PROB_FIELD_NUMBER: _ClassVar[int]
    OAT1_UPTAKE_PROB_FIELD_NUMBER: _ClassVar[int]
    OAT3_UPTAKE_PROB_FIELD_NUMBER: _ClassVar[int]
    OCT2_UPTAKE_PROB_FIELD_NUMBER: _ClassVar[int]
    EFFLUX_SCORE_FIELD_NUMBER: _ClassVar[int]
    UPTAKE_SCORE_FIELD_NUMBER: _ClassVar[int]
    pgp_efflux_prob: float
    bcrp_efflux_prob: float
    mrp2_efflux_prob: float
    oatp1b1_uptake_prob: float
    oatp1b3_uptake_prob: float
    oat1_uptake_prob: float
    oat3_uptake_prob: float
    oct2_uptake_prob: float
    efflux_score: float
    uptake_score: float
    def __init__(self, pgp_efflux_prob: _Optional[float] = ..., bcrp_efflux_prob: _Optional[float] = ..., mrp2_efflux_prob: _Optional[float] = ..., oatp1b1_uptake_prob: _Optional[float] = ..., oatp1b3_uptake_prob: _Optional[float] = ..., oat1_uptake_prob: _Optional[float] = ..., oat3_uptake_prob: _Optional[float] = ..., oct2_uptake_prob: _Optional[float] = ..., efflux_score: _Optional[float] = ..., uptake_score: _Optional[float] = ...) -> None: ...

class TransporterResponse(_message.Message):
    __slots__ = ("success", "predictions", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    predictions: TransporterPredictions
    error: str
    def __init__(self, success: bool = ..., predictions: _Optional[_Union[TransporterPredictions, _Mapping]] = ..., error: _Optional[str] = ...) -> None: ...

class VisualRequest(_message.Message):
    __slots__ = ("smiles", "generate_esp", "generate_fukui", "resolution")
    SMILES_FIELD_NUMBER: _ClassVar[int]
    GENERATE_ESP_FIELD_NUMBER: _ClassVar[int]
    GENERATE_FUKUI_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    generate_esp: bool
    generate_fukui: bool
    resolution: int
    def __init__(self, smiles: _Optional[str] = ..., generate_esp: bool = ..., generate_fukui: bool = ..., resolution: _Optional[int] = ...) -> None: ...

class VisualMaps(_message.Message):
    __slots__ = ("esp_map_png", "fukui_map_png", "resolution", "colormap")
    ESP_MAP_PNG_FIELD_NUMBER: _ClassVar[int]
    FUKUI_MAP_PNG_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_FIELD_NUMBER: _ClassVar[int]
    COLORMAP_FIELD_NUMBER: _ClassVar[int]
    esp_map_png: bytes
    fukui_map_png: bytes
    resolution: int
    colormap: str
    def __init__(self, esp_map_png: _Optional[bytes] = ..., fukui_map_png: _Optional[bytes] = ..., resolution: _Optional[int] = ..., colormap: _Optional[str] = ...) -> None: ...

class VisualMapsResponse(_message.Message):
    __slots__ = ("success", "maps", "error", "computation_time")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MAPS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_TIME_FIELD_NUMBER: _ClassVar[int]
    success: bool
    maps: VisualMaps
    error: str
    computation_time: float
    def __init__(self, success: bool = ..., maps: _Optional[_Union[VisualMaps, _Mapping]] = ..., error: _Optional[str] = ..., computation_time: _Optional[float] = ...) -> None: ...

class ExplainRequest(_message.Message):
    __slots__ = ("smiles", "prediction")
    SMILES_FIELD_NUMBER: _ClassVar[int]
    PREDICTION_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    prediction: PBPKPrediction
    def __init__(self, smiles: _Optional[str] = ..., prediction: _Optional[_Union[PBPKPrediction, _Mapping]] = ...) -> None: ...

class Interpretation(_message.Message):
    __slots__ = ("shap_values", "attention_quantum", "attention_enzyme", "attention_transporter", "attention_visual", "mechanism_summary", "key_insights", "important_atoms", "important_regions")
    class ShapValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class AttentionQuantumEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class AttentionEnzymeEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class AttentionTransporterEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class AttentionVisualEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    SHAP_VALUES_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_QUANTUM_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_ENZYME_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_TRANSPORTER_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_VISUAL_FIELD_NUMBER: _ClassVar[int]
    MECHANISM_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    KEY_INSIGHTS_FIELD_NUMBER: _ClassVar[int]
    IMPORTANT_ATOMS_FIELD_NUMBER: _ClassVar[int]
    IMPORTANT_REGIONS_FIELD_NUMBER: _ClassVar[int]
    shap_values: _containers.ScalarMap[str, float]
    attention_quantum: _containers.ScalarMap[str, float]
    attention_enzyme: _containers.ScalarMap[str, float]
    attention_transporter: _containers.ScalarMap[str, float]
    attention_visual: _containers.ScalarMap[str, float]
    mechanism_summary: str
    key_insights: _containers.RepeatedScalarFieldContainer[str]
    important_atoms: _containers.RepeatedScalarFieldContainer[str]
    important_regions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, shap_values: _Optional[_Mapping[str, float]] = ..., attention_quantum: _Optional[_Mapping[str, float]] = ..., attention_enzyme: _Optional[_Mapping[str, float]] = ..., attention_transporter: _Optional[_Mapping[str, float]] = ..., attention_visual: _Optional[_Mapping[str, float]] = ..., mechanism_summary: _Optional[str] = ..., key_insights: _Optional[_Iterable[str]] = ..., important_atoms: _Optional[_Iterable[str]] = ..., important_regions: _Optional[_Iterable[str]] = ...) -> None: ...

class ExplanationResponse(_message.Message):
    __slots__ = ("success", "interpretation", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    INTERPRETATION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    interpretation: Interpretation
    error: str
    def __init__(self, success: bool = ..., interpretation: _Optional[_Union[Interpretation, _Mapping]] = ..., error: _Optional[str] = ...) -> None: ...

class AttentionRequest(_message.Message):
    __slots__ = ("smiles",)
    SMILES_FIELD_NUMBER: _ClassVar[int]
    smiles: str
    def __init__(self, smiles: _Optional[str] = ...) -> None: ...

class AttentionResponse(_message.Message):
    __slots__ = ("success", "attention_heatmap_png", "attention_weights", "error")
    class AttentionWeightsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_HEATMAP_PNG_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    attention_heatmap_png: bytes
    attention_weights: _containers.ScalarMap[str, float]
    error: str
    def __init__(self, success: bool = ..., attention_heatmap_png: _Optional[bytes] = ..., attention_weights: _Optional[_Mapping[str, float]] = ..., error: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ("component",)
    COMPONENT_FIELD_NUMBER: _ClassVar[int]
    component: str
    def __init__(self, component: _Optional[str] = ...) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "status", "component_status", "metrics", "last_check")
    class ComponentStatusEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    COMPONENT_STATUS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    LAST_CHECK_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    status: str
    component_status: _containers.ScalarMap[str, str]
    metrics: _containers.ScalarMap[str, str]
    last_check: _timestamp_pb2.Timestamp
    def __init__(self, healthy: bool = ..., status: _Optional[str] = ..., component_status: _Optional[_Mapping[str, str]] = ..., metrics: _Optional[_Mapping[str, str]] = ..., last_check: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...
