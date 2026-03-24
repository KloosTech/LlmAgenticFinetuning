"""LangChain tools available to the AgentForge agent."""

from __future__ import annotations

from typing import List


def get_tools(tool_names: List[str]):
    """Return a list of LangChain tools filtered to the requested names."""
    available: dict = {}

    # Web search via DuckDuckGo
    try:
        from langchain_community.tools import DuckDuckGoSearchRun  # type: ignore[import]
        available["web_search"] = DuckDuckGoSearchRun(
            name="web_search",
            description="Search the web for current information. Input should be a search query.",
        )
    except ImportError:
        pass

    # Shell execution
    try:
        from langchain_community.tools import ShellTool  # type: ignore[import]
        shell_tool = ShellTool()
        shell_tool.description = (
            "Run a shell command and return its stdout. "
            "Use for file operations, running scripts, or checking system state. "
            "Input must be a valid shell command string."
        )
        available["shell"] = shell_tool
    except ImportError:
        pass

    # Python REPL
    try:
        from langchain_experimental.tools import PythonREPLTool  # type: ignore[import]
        available["python_repl"] = PythonREPLTool()
    except ImportError:
        # Fallback: simple exec-based tool
        from langchain.tools import Tool
        import io, contextlib

        def _run_python(code: str) -> str:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(code, {})  # noqa: S102
            except Exception as e:
                return f"Error: {e}"
            return buf.getvalue() or "(no output)"

        available["python_repl"] = Tool(
            name="python_repl",
            func=_run_python,
            description=(
                "Execute Python code and return the output. "
                "Useful for calculations, data manipulation, or running logic. "
                "Input should be valid Python code."
            ),
        )

    # Read file
    try:
        from langchain_community.tools import ReadFileTool  # type: ignore[import]
        available["read_file"] = ReadFileTool()
    except ImportError:
        from langchain.tools import Tool

        def _read_file(path: str) -> str:
            try:
                return open(path).read()
            except Exception as e:
                return f"Error reading file: {e}"

        available["read_file"] = Tool(
            name="read_file",
            func=_read_file,
            description="Read the contents of a local file. Input should be a file path.",
        )

    selected = []
    for name in tool_names:
        if name in available:
            selected.append(available[name])
        else:
            import warnings
            warnings.warn(f"Tool '{name}' not available — skipping.", stacklevel=2)

    return selected
