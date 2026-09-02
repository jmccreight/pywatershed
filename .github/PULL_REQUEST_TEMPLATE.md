## Checklist

<!-- Strike through items that aren't relevant to your change, e.g.
     - [ ] ~~Tests added~~ ; keep them in the list rather than deleting. -->

- [ ] Closes #xxxx
- [ ] Tests added
- [ ] Performance benchmarks added or run
- [ ] User visible changes (including notable bug fixes) are documented in `whats-new.rst`
- [ ] The ``(:pull:`XXX`)`` placeholder in `whats-new.rst` is replaced with this PR's number
- [ ] New public classes/functions are exported in `pywatershed/__init__.py` and listed in `doc/api/*.rst`

## Docs

Look for this PR number in our [read-the-docs builds](https://app.readthedocs.org/projects/pywatershed/builds/) where you can browse on-line.

You can alternatively get a CI-build of the docs by entering the PR number for the `###` in https://github.com/DOI-USGS/pywatershed/pull/###/checks then clicking on `Documentation Build` and finally looking for the `documentation-html` artifact which will download as a zip file.
