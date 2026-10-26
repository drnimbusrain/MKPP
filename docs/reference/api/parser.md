# Mechanism & Environment Parser (`mkpp.parser`)

The `mkpp.parser` module parses and validates chemical mechanism and environmental configuration files conforming to the **OpenAtmos v1** specification. Both JSON (`.json`) and YAML (`.yaml`, `.yml`) configuration files are supported with automatic format detection and structural parity.

YAML parsing uses PyYAML's `CSafeLoader` when LibYAML is available, with
`SafeLoader` retained for installations without the C extension. Both loaders
reject Python-specific object construction tags and report malformed YAML as
structured compilation errors. Install a LibYAML-enabled PyYAML build for
large-mechanism parsing performance.

::: mkpp.parser
