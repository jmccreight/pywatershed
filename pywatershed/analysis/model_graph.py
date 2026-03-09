import pathlib as pl
import tempfile

from ..base.conservative_process import ConservativeProcess
from ..base.model import Model
from ..utils.optional_import import import_optional_dependency


class ModelGraph:
    """Visualize a pywatershed Model as a directed graph.

    Creates a GraphViz visualization showing processes, their inputs/variables,
    and the data flow between them. Uses the dot layout engine with left-to-right
    orientation.

    Args:
        model: The pywatershed Model to visualize
        show_params: Whether to display parameters in process nodes
        process_colors: Dictionary mapping process names to colors for node borders
        node_penwidth: Width of node borders (default: 2)
        edge_penwidth: Width of edges/arrows (default: 1.5)
        edge_arrowsize: Size of arrowheads (default: 1.2)
        default_edge_color: Color for edges between processes (default: "black")
        from_file_edge_color: Color for edges from file inputs (default: same as
            default_edge_color)
        hide_variables: If True, hide individual variable names and only show
            process-to-process connections (default: True)

    Example:
        >>> import pywatershed as pws
        >>> model = pws.Model(process_list, control=control, parameters=params)
        >>> mg = pws.analysis.ModelGraph(model, hide_variables=False)
        >>> mg.SVG()
    """

    def __init__(
        self,
        model: Model,
        show_params: bool = False,
        process_colors: dict[str, str] | None = None,
        node_penwidth: int = 2,
        edge_penwidth: float = 1.5,
        edge_arrowsize: float = 1.2,
        default_edge_color: str = "black",
        from_file_edge_color: str | None = None,
        hide_variables: bool = True,
    ) -> None:
        self.pydot = import_optional_dependency("pydot")
        self.graph = None

        self.model = model
        self.show_params = show_params
        self.process_colors = process_colors
        self.node_penwidth = node_penwidth
        self.edge_penwidth = edge_penwidth
        self.edge_arrowsize = edge_arrowsize
        self.default_edge_color = default_edge_color
        if not from_file_edge_color:
            self.from_file_edge_color = default_edge_color
        else:
            self.from_file_edge_color = from_file_edge_color
        self.hide_variables = hide_variables

    def build_graph(self) -> None:
        """Build the GraphViz graph structure.

        Creates nodes for all processes and files, establishes connections
        between them, and sets up invisible ordering edges to enforce
        left-to-right layout according to model.process_order.

        The graph uses the dot layout engine with:
        - Left-to-right orientation (rankdir="LR")
        - Invisible edges to control node ordering
        - constraint=false on data edges so they don't affect layout
        """
        # Build the process nodes
        self.process_nodes = {}
        for process in self.model.process_order:
            self.process_nodes[process] = self._process_node(
                process,
                self.model.processes[process],
                show_params=self.show_params,
            )

        # Build connections between processes and files
        self.files = []
        self.connections = []
        for process in self.model.process_order:
            frm_already = []
            for var, frm in self.model.process_input_from[process].items():
                var_con = f":{var}"

                if self.hide_variables:
                    # When hiding variables, connect directly to process
                    # and avoid duplicate edges from same source
                    var_con = ""
                    if frm in frm_already:
                        continue
                    else:
                        frm_already += [frm]

                if not isinstance(frm, pl.Path):
                    color = self.default_edge_color
                    if self.process_colors:
                        color = self.process_colors[frm]
                    self.connections += [
                        (
                            f"{frm}{var_con}",
                            f"{process}{var_con}",
                            color,
                        )
                    ]

                else:
                    file_name = frm.name
                    self.files += [file_name]
                    self.connections += [
                        (
                            f"Files:{file_name.split('.')[0]}",
                            f"{process}{var_con}",
                            self.from_file_edge_color,
                        )
                    ]

        # Build the file inputs node
        self.file_node = self._file_node(self.files)

        # Create the GraphViz graph with layout settings
        self.graph = self.pydot.Dot(
            graph_type="digraph",
            layout="dot",
            rankdir="LR",
            nodesep="0.5",
            ranksep="1.0",
        )

        # Add an invisible source node to force Files to leftmost position
        source_node = self.pydot.Node(
            "_source_",
            style="invis",
            width="0",
            height="0",
        )
        self.graph.add_node(source_node)

        self.graph.add_node(self.file_node)

        # Add invisible edge from source to Files to control rank
        self.graph.add_edge(
            self.pydot.Edge("_source_", "Files", style="invis")
        )

        for process in self.model.process_order:
            self.graph.add_node(self.process_nodes[process])

        # Add invisible edges to enforce left-to-right process ordering.
        # These edges control the layout while constraint=false on data
        # edges prevents them from affecting node positions.
        if len(self.model.process_order) > 0:
            first_process = self.model.process_order[0]
            self.graph.add_edge(
                self.pydot.Edge(
                    "Files", first_process, style="invis", weight="10"
                )
            )

            for i in range(len(self.model.process_order) - 1):
                from_process = self.model.process_order[i]
                to_process = self.model.process_order[i + 1]
                self.graph.add_edge(
                    self.pydot.Edge(
                        from_process, to_process, style="invis", weight="10"
                    )
                )

        # Add data flow edges with constraint=false so they don't affect layout
        for con in self.connections:
            self.graph.add_edge(
                self.pydot.Edge(
                    con[0],
                    con[1],
                    color=con[2],
                    constraint="false",
                    penwidth=str(self.edge_penwidth),
                    arrowsize=str(self.edge_arrowsize),
                )
            )

    def SVG(self, verbose: bool = False, dpi: int = 45) -> None:
        """Render and display the graph as SVG in a Jupyter notebook.

        Args:
            verbose: If True, print the temporary file path
            dpi: Resolution for the SVG output (default: 45)
        """

        ipdisplay = import_optional_dependency("IPython.display")

        tmp_file = pl.Path(tempfile.NamedTemporaryFile().name)
        if self.graph is None:
            self.build_graph()
        self.graph.write_svg(tmp_file, prog=["dot", f"-Gdpi={dpi}"])
        if verbose:
            print(f"Displaying SVG written to temp file: {tmp_file}")

        ipdisplay.display(ipdisplay.SVG(tmp_file))

    def _process_node(
        self, process_name: str, process, show_params: bool = False
    ):
        """Create a node for a process with its inputs, variables, and parameters.

        Args:
            process_name: Name of the process in the model
            process: The process instance
            show_params: Whether to include parameters in the node

        Returns:
            A pydot.Node with an HTML-like table label showing the process
            structure
        """
        inputs = process.get_inputs()
        variables = process.get_variables()
        params = process.get_parameters()
        cls = process.__class__.__name__

        label = (
            f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="1">\n'  # noqa: E501
            f'    <TR><TD COLSPAN="6">"{process_name}"</TD></TR>\n'
            f'    <TR><TD COLSPAN="6">{cls}</TD></TR>\n'
        )

        category_colors = {
            "inputs": "lightblue",
            "params": "orange",
            "variables": "lightgreen",
        }

        show_categories = {
            "inputs": inputs,
            "params": params,
            "variables": variables,
        }
        if not show_params:
            _ = show_categories.pop("params")

        if isinstance(process, ConservativeProcess):
            mass_budget_vars = [
                var
                for comp, vars in process.get_mass_budget_terms().items()
                for var in vars
            ]
        else:
            mass_budget_vars = []

        for varset_name, varset in show_categories.items():
            n_vars = len(varset)
            if not n_vars:
                continue

            varset_col_span = 2
            if self.hide_variables:
                varset = []
                n_vars = 0
                varset_col_span = 6

            label += "    <TR>\n"
            label += f'        <TD COLSPAN="{varset_col_span}" ROWSPAN="{n_vars + 1}"'  # noqa: E501
            label += f'         BGCOLOR="{category_colors[varset_name]}">{varset_name}</TD>\n'  # noqa: E501
            label += "    </TR>\n"

            for vv in sorted(varset):
                border_color_str = ""
                if vv in mass_budget_vars:
                    border_color_str = 'border="1" COLOR="BLUE"'
                label += "    <TR>\n"
                label += f'        <TD COLSPAN="4" BGCOLOR="{category_colors[varset_name]}" {border_color_str} PORT="{vv}"><FONT POINT-SIZE="9.0">{vv}</FONT></TD>\n'  # noqa: E501
                label += "    </TR>\n"

        label += "</TABLE>>\n"

        color_str = ""
        if self.process_colors:
            color_str = f'"{self.process_colors[process_name]}"'

        node = self.pydot.Node(
            process_name,
            label=label,
            shape="box",
            color=color_str,
            penwidth=f'"{self.node_penwidth}"',
        )

        return node

    def _file_node(self, files: list[str]):
        """Create a node listing all input files.

        Args:
            files: List of file names that provide inputs to the model

        Returns:
            A pydot.Node with an HTML-like table label listing all files
        """
        files = list(set(files))
        label = (
            '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="1">\n'  # noqa
            '    <TR><TD COLSPAN="1">Files</TD></TR>\n'
        )

        for file in files:
            label += "    <TR>\n"
            label += f'        <TD COLSPAN="1"  BGCOLOR="gray50" PORT="{file.split(".")[0]}"><FONT POINT-SIZE="9.0">{file}</FONT></TD>\n'  # noqa
            label += "    </TR>\n"

        label += "</TABLE>>\n"
        node = self.pydot.Node(
            "Files",
            label=label,
            shape="note",
            color=f'"{self.from_file_edge_color}"',
            penwidth=f'"{self.node_penwidth}"',
        )
        return node
