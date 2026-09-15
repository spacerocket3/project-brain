# Security policy

## Reporting

Please report vulnerabilities privately through GitHub Security Advisories for this
repository. Do not open a public issue containing a working exploit, credential, private
repository content, or sensitive filesystem path.

## Trust boundary

Project Brain reads local repositories explicitly registered by the machine owner. The MCP
server cannot register arbitrary paths. Its results may contain source code from those
repositories, so expose the server only to clients authorized to read them.

The HTTP server binds to loopback by default and has no authentication layer. Do not bind it
to a public interface without placing an authenticated, encrypted proxy in front of it.

Project Brain attempts to exclude common secret files, keys, binaries, dependency trees,
and build output. Pattern-based exclusion is defense in depth, not a guarantee that source
files contain no secrets. Review repository hygiene before indexing sensitive code.
