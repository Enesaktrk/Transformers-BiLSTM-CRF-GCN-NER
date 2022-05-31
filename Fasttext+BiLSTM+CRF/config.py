# Some configuration variables.

PAD_IDX = 0 # Padding index
UNK_IDX = 1 # Unknown word index

# Tags for the named entities.
UNIQUE_TAGS = ["B", "I", "O", "PAD"]
'''
UNIQUE_TAGS = [
    "I-Immaterial_anatomical_entity", "I-Organ",
    "B-Organism_substance", "I-Cell", "B-Organism",
    "I-Amino_acid", "B-Tissue", "B-Anatomical_system",
    "B-Cellular_component", "I-Developing_anatomical_structure",
    "B-Pathological_formation", "B-Organism_subdivision", "B-Simple_chemical",
    "B-Immaterial_anatomical_entity", "I-Multi-tissue_structure", "I-Cancer", "I-Pathological_formation",
    "I-Gene_or_gene_product", "B-Multi-tissue_structure", "B-Developing_anatomical_structure", "O",
    "B-Organ", "I-Tissue", "I-Anatomical_system", "I-Cellular_component", "I-Organism", "B-Cell", "B-Cancer", "B-Amino_acid",
    "I-Organism_substance", "I-Organism_subdivision", "I-Simple_chemical", "B-Gene_or_gene_product", "PAD"
]
'''

tag2idx = {tag: idx for idx, tag in enumerate(UNIQUE_TAGS)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}