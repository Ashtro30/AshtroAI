# Information Security Policy

This policy applies to all Meridian Labs employees, contractors, systems and devices.

## Authentication and Access

- Passwords must be at least **14 characters** long and must not be reused across systems.
- Multi-factor authentication (MFA) is mandatory for all production systems, source-code hosting and the cloud console.
- Access follows least privilege and is reviewed every quarter. Access is revoked on the last working day of departing staff.

## Key and Secret Rotation

- API keys and service credentials must be rotated every **90 days**.
- SSH keys must be rotated every 180 days.
- Device keys for deployed IoT hardware must be rotated at least every 90 days.
- Secrets must never be committed to source control; use the approved secrets manager.

## Vulnerability Remediation SLAs

| Severity | Fix within          |
| -------- | ------------------- |
| Critical | 7 days              |
| High     | 30 days             |
| Medium   | 90 days             |
| Low      | Next release cycle  |

Vulnerabilities in internet-facing systems are triaged within 24 hours of discovery.

## Device and Endpoint Security

- Full-disk encryption is required on all laptops.
- Operating-system and browser updates must be installed within 14 days of release.
- Lost or stolen devices must be reported to the security team within 2 hours.

## Incident Reporting

Suspected security incidents are reported to security@meridianlabs.example immediately and handled as SEV1 incidents under the Incident Response Runbook until triaged.
