# Incident Response Runbook

This runbook defines how Meridian Labs classifies, responds to, and learns from production incidents.

## Severity Levels

- **SEV1 - Critical:** complete outage of a customer-facing service, confirmed data breach, or safety-relevant device behaviour. First response within **15 minutes**, 24x7. An Incident Commander is assigned immediately and status updates are posted every 30 minutes.
- **SEV2 - Major:** a major feature is degraded or a subset of customers is impacted. First response within **1 hour** during business hours. Status updates every 2 hours.
- **SEV3 - Minor:** cosmetic or low-impact issue with a workaround. Handled by the next business day through the normal ticket queue.

## Incident Roles

- **Incident Commander (IC):** owns decisions and coordination; does not debug.
- **Ops Lead:** performs diagnosis and mitigation.
- **Communications Lead:** updates the status page and stakeholders.
- **Scribe:** keeps a timestamped log in the incident channel.

## SEV1 Response Procedure

1. Declare the incident in the #incidents channel and page the on-call engineer.
2. Assign the Incident Commander, Ops Lead, Communications Lead and Scribe.
3. Open a video bridge and link it in the incident channel.
4. Mitigate first: roll back the most recent deployment before investigating root cause.
5. Publish a status-page update within 30 minutes and then every 30 minutes until resolved.
6. Confirm recovery with monitoring, then close the incident and notify stakeholders.
7. Schedule a blameless postmortem.

## Rollback Procedure

Use the deployment tool to redeploy the previous tagged release. Verify health checks are green for 10 consecutive minutes before declaring the rollback successful. Database migrations must be backward compatible so a rollback never requires a schema change.

## Postmortems

A written, blameless postmortem is due within **5 business days** of any SEV1 or SEV2 incident. It must include a timeline, root cause, customer impact, what went well, what went poorly, and tracked action items with owners and due dates.
