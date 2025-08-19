# Versioning Strategy

Elelem uses **Git tags** for version management with `setuptools-scm`. No manual version updates needed!

## How It Works

1. **Automatic Version Detection**: Version is automatically determined from Git tags
2. **No Manual Updates**: Never edit version numbers in code
3. **Dynamic Import**: `elelem.__version__` always reflects the current version

## Creating a New Release

### 1. Tag the Release
```bash
# Create annotated tag (recommended)
git tag -a v1.0.0 -m "Release v1.0.0: Description of changes"

# Push tag to GitHub
git push origin v1.0.0
```

### 2. Automatic Actions
When you push a tag starting with `v`:
- GitHub Actions builds the package
- Creates a GitHub Release with artifacts
- Version is automatically set in the package

## Version Formats

### Tagged Releases
- `v1.0.0` → Package version: `1.0.0`
- `v2.1.3` → Package version: `2.1.3`

### Development Versions
Between tags, version includes commit info:
- `1.0.1.dev5+g1234567` (5 commits after v1.0.0, commit hash g1234567)

## Installation Methods

### From Git Tag
```bash
# Install specific version
pip install git+https://github.com/guybrush1984/Elelem.git@v1.0.0

# Install latest tag
pip install git+https://github.com/guybrush1984/Elelem.git

# Install development version
pip install git+https://github.com/guybrush1984/Elelem.git@main
```

### SSH Installation (Private Repo)
```bash
# Using SSH key
pip install git+ssh://git@github.com/guybrush1984/Elelem.git@v1.0.0
```

## Checking Version

### In Python
```python
import elelem
print(elelem.__version__)  # Shows current version
```

### From Git
```bash
git describe --tags  # Shows current version based on tags
```

## Best Practices

1. **Use Semantic Versioning**: `MAJOR.MINOR.PATCH`
   - MAJOR: Breaking changes
   - MINOR: New features (backwards compatible)
   - PATCH: Bug fixes

2. **Tag Messages**: Always use descriptive messages
   ```bash
   git tag -a v1.1.0 -m "Add support for new provider X"
   ```

3. **Pre-releases**: Use suffixes for pre-release versions
   ```bash
   git tag -a v2.0.0-rc1 -m "Release candidate 1 for v2.0.0"
   ```

## Troubleshooting

### Version Shows as "unknown"
- Not installed in development mode: `pip install -e .`
- Git repository not available (e.g., installed from ZIP)

### Version Not Updating
- Ensure Git tags are fetched: `git fetch --tags`
- Reinstall package: `pip install --force-reinstall -e .`

## Implementation Details

- Version determined by `setuptools-scm` at build time
- `_version.py` auto-generated (git-ignored)
- Falls back to "unknown" if Git info unavailable