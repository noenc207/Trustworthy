"""Experiment lineage graph generator."""

import xml.etree.ElementTree as ET

from .experiment_registry import ExperimentRegistry


class LineageGraph:
    """Generates a lineage graph for experiments."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        """Initializes the lineage graph.

        Args:
            registry (ExperimentRegistry): The experiment registry.
        """
        self.registry = registry

    def export_graphml(self, filepath: str) -> None:
        """Exports the lineage graph to a GraphML file.

        Args:
            filepath (str): The path to save the GraphML file.
        """
        graphml = ET.Element(
            "graphml",
            xmlns="http://graphml.graphdrawing.org/xmlns",
        )

        # Add keys for node attributes
        ET.SubElement(graphml, "key", id="name", **{"for": "node", "attr.name": "name", "attr.type": "string"})
        ET.SubElement(graphml, "key", id="seed", **{"for": "node", "attr.name": "seed", "attr.type": "string"})
        ET.SubElement(graphml, "key", id="commit", **{"for": "node", "attr.name": "commit", "attr.type": "string"})
        ET.SubElement(graphml, "key", id="hash", **{"for": "node", "attr.name": "hash", "attr.type": "string"})

        graph = ET.SubElement(graphml, "graph", id="G", edgedefault="directed")

        experiments = self.registry.get_all()

        for exp in experiments:
            node = ET.SubElement(graph, "node", id=exp.id)

            data_name = ET.SubElement(node, "data", key="name")
            data_name.text = exp.name

            data_seed = ET.SubElement(node, "data", key="seed")
            data_seed.text = str(exp.random_seed) if exp.random_seed is not None else ""

            data_commit = ET.SubElement(node, "data", key="commit")
            data_commit.text = exp.git_commit if exp.git_commit else ""

            data_hash = ET.SubElement(node, "data", key="hash")
            data_hash.text = exp.config_hash

            for parent_id in exp.parents:
                ET.SubElement(
                    graph,
                    "edge",
                    source=parent_id,
                    target=exp.id,
                )

        tree = ET.ElementTree(graphml)
        tree.write(filepath, encoding="utf-8", xml_declaration=True)
