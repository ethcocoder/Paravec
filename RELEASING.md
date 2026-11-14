# Release Checklist

Paravec is currently a research prototype. Do not describe it as production-ready until the following are complete:

- CI passes on supported operating systems and Python versions.
- Wheel and source distribution build and install in a clean environment.
- Equal-total-byte comparisons include the uniform scalar baseline and strong PQ/OPQ baselines on representative datasets.
- Recall, build/storage overhead, and rerank behavior are reported across repeated seeds and at least one out-of-distribution dataset.
- A project license is selected by the maintainer and included before distributing a formal release.
- README claims are limited to measured results; synthetic test results are not presented as benchmark results.

## Local checks

```powershell
python -m unittest discover -s tests -v
python -m pip install build
python -m build
```

Once the checklist is satisfied, create a version tag and a GitHub Release with the built wheel and source distribution attached. Do not publish to PyPI until package ownership, naming, and release credentials are intentionally configured.