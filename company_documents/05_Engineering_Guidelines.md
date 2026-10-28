# Nexora Technologies Pvt. Ltd.

## Engineering Guidelines and Operations Handbook

| Field | Detail |
|---|---|
| Document ID | NXR-ENG-001 |
| Version | 3.5 |
| Effective Date | 1 April 2026 |
| Next Scheduled Review | 31 March 2027 |
| Document Owner | Chief Technology Officer (CTO), Kavita Rao |
| Co-Owner | Vice President of Engineering, Vikram Nair |
| Security Reviewer | Chief Information Security Officer (CISO), Rohan Kulkarni |
| Approved By | Chief Executive Officer (CEO), Arjun Mehta |
| Classification | Internal |

---

## 1. Purpose

This handbook describes how Nexora Technologies Pvt. Ltd. ("Nexora", "the Company") designs, builds, tests, releases, and operates software. It gives engineers a shared standard for quality, reliability, and security, so that customers can depend on Nexora products and so that engineers can move between teams without relearning basic practices.

This handbook applies the security rules of the Information Technology Security Policy (NXR-IT-001) to engineering work. Where this handbook and NXR-IT-001 differ on access to systems or handling of data, **NXR-IT-001 prevails**. In particular, every rule about production access in Section 8 of this handbook is an application of Section 6 of NXR-IT-001, and must not be read as relaxing it.

The handbook also relies on the Human Resources Policy Manual (NXR-HR-001) for probation, on-call allowances, and separation; on the Remote Work and Hybrid Working Policy (NXR-HR-002) for working remotely; on the Travel and Expense Reimbursement Policy (NXR-FIN-001) for conference and travel spending; and on the Code of Conduct and Ethics Policy (NXR-COC-001) for professional standards.

## 2. Scope

2.1 This handbook applies to all employees, contractors, and interns who write, review, deploy, operate, or support Nexora software, in the Engineering, Platform, Site Reliability Engineering (SRE), Quality Engineering, and Data Engineering teams at the Pune, Bengaluru, and Hyderabad offices.

2.2 It applies to all Nexora-owned services and repositories, including customer-facing products, internal tools, infrastructure as code, and data pipelines.

2.3 It applies to every environment: development, staging, and production. Requirements that apply only to production are stated as such.

## 3. Roles and Responsibilities

| Role | Responsibility |
|---|---|
| CTO | Owns technical strategy and this handbook |
| VP of Engineering | Accountable for delivery, reliability, and compliance in Engineering; co-approves privileged production access |
| Engineering Manager (EM), grade M1 | Manages a team; approves code-owner assignments, change requests, and access requests within the team; conducts quarterly access reviews |
| Tech Lead / Staff Engineer, grade L5 | Leads technical decisions for a service or area; is a default code owner |
| Engineer (L1 – L4) | Writes, reviews, and operates code; follows this handbook |
| SRE and Platform teams | Operate the deployment pipeline, production infrastructure, and Nexora Access Vault (NAV) integrations |
| IT & Security | Provides security tooling, security approvers, and incident support |
| Incident Commander | The person who leads the response to a SEV1 or SEV2 incident (Section 10) |

Department Head for Engineering, as used in all Nexora policies, means the VP of Engineering.

## 4. Engineering Onboarding and Secure Development Training

4.1 **Onboarding plan.** Each new engineer receives a 30-60-90 day onboarding plan from their Engineering Manager, containing learning goals, a buddy, and a first production-impacting project by day 45. The plan is reviewed at the probation check-ins described in Section 5.2 of NXR-HR-001.

4.2 **Required training.** Before being given any of the rights below, engineers must complete the following training on the NexoraHub learning platform:

| Training | Required Before | Owner |
|---|---|---|
| Information Security Awareness | Any access to Nexora systems beyond email (within 10 working days of joining) | CISO |
| Secure Development Fundamentals | Merge rights on any repository | CISO and CTO |
| Production Access and Change Management | Any production access above R1 | SRE |
| On-Call Readiness | Joining an on-call rotation | SRE |

4.3 **Merge rights for engineers on probation.** An engineer who is on probation (within the first 6 months of employment, including any extension) may receive merge rights after completing Secure Development Fundamentals. Their pull requests are subject to the review rules in Section 5.4.

4.4 **Annual refresher.** All engineers must complete the annual security and secure coding refresher by the last working day of June, as required by Section 13 of NXR-IT-001.

4.5 **Conferences and learning.** Engineers may attend conferences and training with their manager's approval, using the annual learning budget of ₹40,000 (NXR-HR-001, Section 8.3). Where travel is involved, expenses are approved and reimbursed under NXR-FIN-001, and any trip above ₹25,000 requires the Department Head (the VP of Engineering) and the Finance approver.

## 5. Source Control, Branching, and Code Review

5.1 **Repositories.** All source code is stored in the Company's approved source control platform. Code must not be stored in personal repositories or on personal devices. Source code is classified as Confidential under Section 4 of NXR-IT-001.

5.2 **Branching model.** Nexora uses trunk-based development with short-lived feature branches:

- The `main` branch is always deployable and is protected. Direct pushes to `main` are disabled.
- Feature branches should live for no longer than 3 working days and be merged through pull requests.
- Branch names follow the pattern `type/TICKET-ID-short-description`, for example `feat/PLAT-482-rate-limit-api`.
- Hotfix branches are used only under the emergency change process in Section 7.6.

5.3 **Commits.** Commits must be signed. Commit messages should follow the Conventional Commits format and refer to the ticket identifier. Pull requests should generally be under 400 lines of change. Larger changes should be split unless there is a strong reason.

5.4 **Code review rules.**

(a) **Production services** (any service that is deployed to production and serves customers or handles customer data): every pull request needs **two approvals**, at least one of which must come from a designated code owner. The author cannot approve their own pull request.

(b) **Approvals from engineers on probation do not count** towards the two required approvals, although their comments are encouraged and valued. At least one approver must be a confirmed engineer at grade L4 or above.

(c) **Internal tools, documentation, and non-production experiments:** one approval is sufficient.

(d) **Security-sensitive changes** (authentication, authorisation, encryption, payment, and access control code) additionally require review by a member of the Security Champions group.

(e) **Infrastructure as code** changes to production networking, identity, or data stores require review by the SRE or Platform team.

5.5 **Review standards.** Reviewers should check correctness, readability, tests, security, performance, and operability. Reviewers should respond within 1 working day. Comments must be respectful and specific, in line with the Code of Conduct (NXR-COC-001).

5.6 **Continuous integration.** Each pull request must pass the CI pipeline, including build, unit tests, linting, static application security testing (SAST), dependency scanning, secret scanning, and licence scanning, before it can be merged. Failing checks must not be bypassed.

5.7 **Code owners.** Every repository has a CODEOWNERS file, maintained by the Engineering Manager, listing individuals or groups responsible for each part of the codebase. Only confirmed engineers may be code owners.

## 6. Testing, Quality, and Vulnerability Management

6.1 **Test pyramid.** Teams should maintain a balance of unit tests, integration tests, contract tests, and a small number of end-to-end tests. New code should have at least 80 percent line coverage, and bug fixes must include a regression test.

6.2 **Non-functional testing.** Services must have performance and load tests before launch and before major changes to traffic-sensitive components. Recovery and failover tests are conducted at least twice a year for services classified as Tier 1 (customer-facing, revenue-critical).

6.3 **Security testing.** In addition to SAST and dependency scanning in CI, Nexora carries out dynamic application security testing on staging before each major release. An independent penetration test of all Tier 1 services is carried out at least once a year by an approved third party, coordinated by IT & Security.

6.4 **Vulnerability remediation.** Vulnerabilities found in code, dependencies, or infrastructure must be fixed in line with Section 8.6 of NXR-IT-001:

| Severity | Maximum Time to Remediate |
|---|---|
| Critical | 7 days |
| High | 30 days |
| Medium | 90 days |
| Low | Next planned maintenance |

Where remediation is not possible in time, the service owner must obtain a time-bound exception from the CISO under Section 14 of NXR-IT-001.

## 7. Release and Change Management

7.1 **Environments.** Code flows through development, staging, and production. Staging must mirror production configuration as closely as is practical, and must not contain unmasked Restricted data (Section 12).

7.2 **Change record.** Every production change requires a change record in the change management system, with an identifier of the form CHG-##### . The record states what is changing, why, the risk, the rollback plan, the testing done, and who will carry out the change.

7.3 **Change types.**

| Type | Description | Approval |
|---|---|---|
| Standard | Pre-approved, low-risk, repeatable changes (for example, routine configuration updates) | Automated, based on template |
| Normal | Planned changes to production | Engineering Manager, plus the Change Advisory Board (CAB) for high-risk changes |
| Emergency | Urgent changes needed to resolve a SEV1 or SEV2 incident or a critical vulnerability | Incident Commander, with retrospective CAB review within 2 working days |

7.4 **Change Advisory Board (CAB).** The CAB meets every Monday and Thursday, and is chaired by the VP of Engineering's delegate. It includes SRE, Security, Quality Engineering, and Customer Success representatives. High-risk changes, for example changes that alter data schemas, authentication, or network configuration, must be reviewed by the CAB before they are deployed.

7.5 **Release windows and freezes.** Planned production deployments are made on **Tuesday, Wednesday, and Thursday between 10:00 and 16:00 IST**. Deployments on Friday, weekends, and public holidays require the approval of the Engineering Manager and the on-call SRE. There is a change freeze during the last 3 business days of each calendar quarter, and during the period from 20 December to 2 January, except for emergency changes.

7.6 **Emergency changes.** Emergency changes follow an abbreviated process but must have a change record, must be linked to an incident record, and must be recorded in NAV. The Incident Commander approves the change, and the Engineering Manager reviews it within 2 working days.

7.7 **Progressive delivery.** Changes should be released gradually, using feature flags, canary releases, or blue-green deployments, with automated rollback if error rate or latency thresholds are breached.

## 8. Production Access and Operations

8.1 **Governing rule.** Access to production is governed by Section 6 of NXR-IT-001. Engineers must not seek to access production in any way other than through Nexora Access Vault (NAV), and must have a valid grant, an approved change record or incident record, and the correct level of access for the task. The five access levels (R0 to R4) are defined in Section 6.2 of NXR-IT-001:

| Level | Meaning | Maximum Grant | Typical Use |
|---|---|---|---|
| R1 | Read-only, dashboards and masked logs | Standing, reviewed quarterly | Debugging, monitoring |
| R2 | Read of unmasked logs or data | 8 hours | Investigating customer-specific issues |
| R3 | Write: deploy, configure, modify data | 4 hours | Releases, planned changes |
| R4 | Privileged / administrative | 2 hours | Infrastructure and database administration |

8.2 **Engineers on probation.** An engineer who is on probation holds **R1 access only**, from their 30th day of employment and after completing the training in Section 4.2. The engineer may shadow R3 and R4 work under the supervision of a confirmed engineer who holds the grant, but must not use the grant. Access above R1 for an engineer on probation is exceptional, requires the **joint written approval of the VP of Engineering and the CISO**, and is recorded in the exceptions register, in line with Section 6.3 of NXR-IT-001. Contractors and interns are limited to R1 access for a maximum of 90 days per grant.

8.3 **Requesting access.** To request a grant, the engineer opens a request in NAV that includes the access level, a duration no longer than the maximum, the purpose, and the change ID (CHG-#####) or incident ID (INC-####) that the work relates to. Approvals are as follows:

- R2: the Engineering Manager and the data owner.
- R3: the Engineering Manager and an on-duty Security approver.
- R4: the VP of Engineering and the CISO.

8.4 **Working practices.**

- All production work is done through NAV sessions, which are recorded for R3 and R4.
- Access must be used only for the stated purpose and stopped when the task is complete. Unused grants should be closed early.
- For destructive or irreversible operations (for example, deleting data or dropping tables) at R4, a second engineer must be present and must review the command before it is run (the two-person rule).
- Ad hoc data changes must be scripted, reviewed in a pull request, and tested in staging before they are run in production, unless an incident requires otherwise.
- Engineers must not copy production data out of the production environment, must not view customer data without a business need, and must not share credentials or NAV sessions.

8.5 **Break-glass access.** During a SEV1 or SEV2 incident, the Incident Commander may grant break-glass access to an on-call engineer through NAV, in line with Section 6.5 of NXR-IT-001. Break-glass grants last for up to 2 hours, notify the CISO and the VP of Engineering automatically, and are reviewed by IT & Security within one working day.

8.6 **Remote access to production.** Production access from outside a Nexora office network is permitted only through NexoraVPN, on an approved device, and with MFA, as set out in Section 6.8 of NXR-HR-002 and Section 8 of NXR-IT-001. Approval to work remotely requires the reporting manager's approval and a working VPN, as set out in Section 5.1 of NXR-HR-002. Production access from outside India is disabled unless approved by the CISO under Section 6.9 of NXR-IT-001.

8.7 **Quarterly review.** Each Engineering Manager must certify their team's production access every quarter within 15 working days, as required by Section 6.8 of NXR-IT-001. Engineers who no longer need access must have it removed immediately.

## 9. On-Call

9.1 **Purpose.** On-call rotations ensure that production services are supported outside working hours. On-call is organised per service group. Each rotation has a primary and a secondary engineer, and rotates weekly, handing over every Monday at 11:00 IST.

9.2 **Eligibility.** To be on the primary or secondary on-call rotation, an engineer must:

- Have been **confirmed** in their role, that is, must have completed the 6-month probation period described in Section 5 of NXR-HR-001 (engineers on probation are not eligible).
- Have completed On-Call Readiness training and shadowed at least two on-call weeks.
- Be eligible for at least R3 access to the services covered, as described in Section 8.
- Have a suitable and reliable connection to the Company's systems under Section 10.5 of NXR-HR-002.

9.3 **Compensation.** Confirmed employees on on-call rotations are paid an on-call allowance of **₹8,000 per week for primary** and **₹3,000 per week for secondary** duty, in line with Section 8.4 of NXR-HR-001. An engineer who works for more than 3 hours outside working hours on a SEV1 or SEV2 incident receives one day of compensatory off, as set out in the same section. The Engineering Manager records incident hours in the incident record.

9.4 **Response times.** The on-call engineer must acknowledge pages within 15 minutes for SEV1 and 30 minutes for SEV2, and must begin work on the incident immediately. If the primary does not acknowledge within the time limit, the alert escalates to the secondary, then to the Engineering Manager, and finally to the VP of Engineering.

9.5 **Limits and wellbeing.** No engineer may be primary on-call for more than one week in any four-week period. Engineers who work through the night on an incident should have at least 11 hours of rest before resuming normal work. Engineering Managers must monitor on-call load and address alert fatigue.

9.6 **Remote on-call.** On-call engineers who work remotely need a stable network connection and a backup connection. Reasonable costs for a backup connection of up to ₹1,000 per month are reimbursed through NexoraExpense, as set out in Section 10.5 of NXR-HR-002.

9.7 **Swaps and leave.** Engineers may swap on-call weeks with a colleague who is also eligible, with the Engineering Manager's approval and record in the scheduling tool. On-call duties must not be scheduled during approved leave under Section 7 of NXR-HR-001.

9.8 **Handover.** At the end of each rotation, the outgoing engineer records open issues, alerts, and follow-ups in the handover log. Engineers who are about to leave the company must transfer their on-call responsibilities and runbook ownership during their notice period, as required by Section 10.3 of NXR-HR-001.

## 10. Incident Response and Post-Incident Reviews

10.1 **Severity levels.** Engineering incidents are classified by impact. The engineering severity levels correspond to the security priority levels in Section 9.3 of NXR-IT-001.

| Severity | Security Priority | Definition | Acknowledge Within | Update Cadence |
|---|---|---|---|---|
| SEV1 | P1 | Major outage of a Tier 1 service, or confirmed breach of Restricted data | 15 minutes | Every 30 minutes |
| SEV2 | P2 | Significant degradation or partial outage; serious security compromise | 30 minutes | Every 60 minutes |
| SEV3 | P3 | Limited impact; workaround exists | 4 business hours | Daily |
| SEV4 | P4 | Minor issue, no customer impact | 1 business day | As needed |

10.2 **Roles.** For SEV1 and SEV2 incidents, the on-call engineer opens an incident record (INC-####) and an Incident Commander is assigned, usually the on-call Engineering Manager. The Incident Commander appoints an Operations Lead, a Communications Lead, and a scribe, and can call on subject matter experts.

10.3 **Security incidents.** Any incident that may involve unauthorised access, data exposure, or malicious activity must be reported to security@nexoratech.example within **one hour**, as required by Section 9.2 of NXR-IT-001. The CISO's Security Incident Response Team (SIRT) then leads the security aspects of the response, and the Incident Commander leads service restoration.

10.4 **Communication.** The Communications Lead updates the internal incident channel and status page at the cadence in Section 10.1. Customer communications are made by Customer Success with Legal & Compliance approval where legal or contractual obligations are involved. Engineers must not communicate externally about incidents.

10.5 **Post-incident review.** A blameless post-incident review is required for every SEV1 and SEV2 incident, and for any incident that involved break-glass access. The review is completed within **5 business days** of resolution, and includes a timeline, root cause, impact, what went well, what went badly, and corrective actions with owners and due dates. SEV1 reviews are presented to the CTO. Actions are tracked in the engineering backlog until closed.

10.6 **Learning.** Post-incident reviews are shared with all Engineering staff. Reviews focus on system and process improvements rather than individual blame.

## 11. Secrets and Credentials Management

11.1 **No secrets in code.** Secrets (passwords, API keys, tokens, certificates, and encryption keys) are Restricted information under Section 4 of NXR-IT-001. They must never be committed to repositories, embedded in container images, pasted in chat or tickets, or stored in documents.

11.2 **Secrets manager.** Secrets must be stored in the Nexora Secrets Manager and delivered to workloads at runtime. Human access to secrets requires an appropriate NAV grant.

11.3 **Rotation.** Service credentials and API keys must be rotated at least every 90 days. Encryption keys are rotated at least annually. Secrets must be rotated immediately if they are suspected to have been exposed, and the exposure must be reported under Section 10.3.

11.4 **Secret scanning.** Secret scanning is enforced in CI and on developer machines. A committed secret is treated as compromised even if removed from history, and must be rotated.

11.5 **Departure of engineers.** When an engineer who had R3 or R4 access, or who knew shared secrets, leaves Nexora, the owning team must rotate the relevant secrets within 5 working days of separation, as required by Section 12.4 of NXR-IT-001. The departing engineer must list the secrets and systems they own as part of the handover under Section 10.3 of NXR-HR-001.

## 12. Data Handling and Environments

12.1 **Classification.** Engineers must handle data according to the classification in Section 4 of NXR-IT-001. Customer data and personal data are Restricted.

12.2 **Production data in lower environments.** Production data must not be copied to development, test, or staging environments unless it has been masked or anonymised using the approved masking tooling, and the masking has been verified by the service owner. Synthetic data should be used wherever possible.

12.3 **Encryption.** Data must be encrypted in transit using TLS 1.2 or higher, and at rest using AES-256 or an equivalent standard. Encryption keys must be managed in the approved key management service.

12.4 **Data residency and retention.** Customer data for Indian customers is stored in Indian cloud regions unless the customer agrees otherwise. Data retention and deletion follow the schedules in the customer contract and the Company's data retention standard, as advised by the DPO, Nisha Patil.

12.5 **Privacy by design.** Engineers must build privacy into products in line with the Digital Personal Data Protection Act, 2023, and Section 9 of NXR-COC-001. New features that process personal data require a privacy review by the DPO's team before launch.

12.6 **Logging.** Application logs must not contain passwords, tokens, or unmasked personal data. Logs are retained in line with Section 11 of NXR-IT-001.

## 13. Open Source and Third-Party Components

13.1 **Permitted licences.** Components under permissive licences (MIT, BSD, Apache-2.0) may be used. Components under copyleft licences (GPL, LGPL, AGPL) or with unclear licences require the approval of Legal & Compliance before use in a product.

13.2 **Software bill of materials.** Each service must generate a software bill of materials (SBOM) at build time, which is used for vulnerability and licence tracking.

13.3 **Dependency hygiene.** Dependencies must be pinned, kept up to date, and scanned in CI. Abandoned or unmaintained packages must be replaced.

13.4 **Third-party services.** Use of any third-party SaaS or API that will process Nexora or customer data requires a vendor assessment under Section 10 of NXR-IT-001. Purchases above ₹25,000 also require approvals under Section 4 of NXR-FIN-001.

13.5 **Contributing to open source.** Engineers may contribute to open-source projects on Company time only with approval from their Engineering Manager and Legal & Compliance. Contributions must not include Confidential information or code. Personal open-source work is subject to the intellectual property terms of the appointment letter and Section 8 of NXR-COC-001.

13.6 **Generative AI tools.** Engineers must use only approved generative AI coding tools and must not paste Confidential or Restricted information, source code, or customer data into unapproved tools, as stated in Section 10.6 of NXR-IT-001. Code produced with AI assistance must be reviewed as any other code under Section 5.4.

## 14. Documentation and Architecture Decisions

14.1 **Runbooks.** Every service with an on-call rotation must have a runbook covering architecture, dependencies, alerts, common failure modes, rollback steps, and escalation contacts. Runbooks are reviewed at least every 6 months and after each SEV1 or SEV2 incident.

14.2 **Architecture decision records.** Significant technical decisions must be documented in an Architecture Decision Record (ADR) stored in the service repository, and reviewed by a Tech Lead. Decisions that change security posture require review by the CISO's team.

14.3 **Service catalogue.** All services must be registered in the service catalogue, with owner, tier, on-call rotation, data classification, and dependencies.

## 15. Compliance and Exceptions

15.1 Compliance with this handbook is monitored through automated checks in CI, quarterly audits by SRE and Security, and the quarterly access reviews under Section 8.7.

15.2 Exceptions to this handbook must be approved by the VP of Engineering and recorded. Exceptions related to security controls require the additional approval of the CISO under Section 14 of NXR-IT-001. An exception may not authorise a breach of NXR-IT-001 Section 6 without the CISO's written approval.

15.3 Breaches of this handbook may result in disciplinary action under Section 11 of NXR-HR-001. Deliberate bypassing of production access controls is gross misconduct.

## 16. Related Documents

- NXR-IT-001 Information Technology Security Policy (Sections 4, 6, 8, 9, 10, 11, 12, 13, and 14)
- NXR-HR-001 Human Resources Policy Manual (Sections 5, 7, 8.4, 10, and 11)
- NXR-HR-002 Remote Work and Hybrid Working Policy (Sections 5, 6, and 10)
- NXR-FIN-001 Travel and Expense Reimbursement Policy (Section 4)
- NXR-COC-001 Code of Conduct and Ethics Policy (Sections 8, 9, and 10)

## 17. Document Control

| Version | Date | Description | Approved By |
|---|---|---|---|
| 1.0 | 1 April 2021 | Initial release | CEO |
| 2.0 | 1 April 2023 | Introduced trunk-based development and CAB | CEO |
| 3.0 | 1 April 2025 | Aligned with NXR-IT-001 access levels R0 to R4 | CEO |
| 3.5 | 1 April 2026 | Clarified probation limits, on-call eligibility, and allowances | CEO |
