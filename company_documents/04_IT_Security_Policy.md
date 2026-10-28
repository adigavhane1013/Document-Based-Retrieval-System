# Nexora Technologies Pvt. Ltd.

## Information Technology Security Policy

| Field | Detail |
|---|---|
| Document ID | NXR-IT-001 |
| Version | 5.0 |
| Effective Date | 1 April 2026 |
| Next Scheduled Review | 31 March 2027 |
| Document Owner | Chief Information Security Officer (CISO), Rohan Kulkarni |
| Approved By | Chief Executive Officer (CEO), Arjun Mehta |
| Co-Reviewer | Chief Technology Officer (CTO), Kavita Rao |
| Classification | Internal |

---

## 1. Purpose

Nexora Technologies Pvt. Ltd. ("Nexora", "the Company") builds and operates software used by customers who trust it with their data. This policy sets the mandatory rules for protecting the Company's information, systems, and customers' data against unauthorised access, loss, and disclosure. It is the authoritative source for all access decisions, technical security controls, and security incident handling.

This policy is supported by, and must be read with:

- NXR-HR-001 Human Resources Policy Manual, which covers onboarding, probation, discipline, and separation.
- NXR-HR-002 Remote Work and Hybrid Working Policy, which sets out who may work remotely and the conditions under which they may do so.
- NXR-ENG-001 Engineering Guidelines and Operations Handbook, which applies these rules to the software development lifecycle and to production operations.
- NXR-FIN-001 Travel and Expense Reimbursement Policy, which covers spending, including on software and travel.
- NXR-COC-001 Code of Conduct and Ethics Policy, which sets expectations on confidentiality, data privacy, and reporting.

## 2. Scope

2.1 This policy applies to all employees (including those on probation), contractors, interns, consultants, and third parties who access Nexora systems or information, at any location and on any device.

2.2 It applies to all information assets: corporate systems (email, collaboration tools, NexoraPeople, NexoraExpense, NexoraHub), source code repositories, build and deployment pipelines, cloud environments, production systems, laptops and mobile devices, and any customer data that Nexora holds or processes.

2.3 The rules apply to all environments. The production environment, which hosts customer-facing services and customer data, receives the highest level of protection and is subject to Section 6 of this policy and Section 8 of NXR-ENG-001.

## 3. Roles and Responsibilities

| Role | Responsibility |
|---|---|
| CEO | Ultimate accountability for information security; approves this policy |
| CISO | Owns this policy; approves exceptions; leads the Security Incident Response Team (SIRT); reports to the CEO and to the Board's Audit Committee |
| CTO | Ensures engineering systems comply; co-owns technical controls |
| VP of Engineering, Vikram Nair | Ensures engineering teams follow Section 6 and NXR-ENG-001; co-approves privileged access |
| Data Protection Officer (DPO), Nisha Patil | Advises on personal data and privacy, and on notification of data breaches |
| Head of Legal & Compliance, Ananya Bhatt | Advises on regulatory obligations and contracts |
| IT & Security team | Operates identity, endpoint, network, and monitoring tools |
| Department Heads and Managers | Ensure their teams follow this policy; review access quarterly |
| All users | Follow this policy, complete training, and report incidents within one hour |

## 4. Data Classification

4.1 All information is classified into one of four levels. The owner of the information is responsible for assigning the classification. If unsure, treat the information as Confidential until advised otherwise.

| Level | Definition | Examples | Handling |
|---|---|---|---|
| Public | Approved for public release | Marketing material, published blog posts, job adverts | No restrictions |
| Internal | For use by Nexora personnel only | Internal policies, org charts, meeting notes | Share only on approved systems; use NexoraVPN when remote |
| Confidential | Sensitive business information whose disclosure would cause harm | Source code, architecture diagrams, unreleased roadmaps, contracts, financial results | Access on a need-to-know basis; encrypt in transit and at rest; no personal devices |
| Restricted | Highest sensitivity; legal, contractual, or severe harm if disclosed | Customer data, personal data (including employee records on NexoraPeople), credentials, encryption keys, authentication secrets | Strict access control; encryption; logged access; never on personal devices or printed at home |

4.2 **Customer and personal data.** Customer data and personal data are Restricted. Handling of personal data must also follow the Digital Personal Data Protection Act, 2023, and Section 9 of NXR-COC-001.

4.3 **Production data in non-production environments.** Restricted data from production must not be copied to development, test, or staging environments unless it has been irreversibly masked or anonymised, as set out in Section 12 of NXR-ENG-001.

4.4 **Labelling.** Documents classified Confidential or Restricted must carry a visible classification label. Emails containing Restricted information must be sent only to those with a need to know and only through the Company's approved encrypted email gateway.

## 5. Identity and Access Control

5.1 **Principles.** Access is granted on the principles of least privilege (the minimum access necessary for the person's role) and need to know. Access is personal. Sharing accounts or credentials is prohibited.

5.2 **Nexora SSO and MFA.** All corporate systems must be accessed through Nexora SSO. Multi-factor authentication (MFA) is mandatory for every user, on every login from outside a Nexora office network, and for all access to production, source code, and cloud consoles regardless of location. SMS-based codes are not permitted for privileged accounts, which must use hardware security keys or authenticator applications.

5.3 **Password standards.** Passwords must be at least **14 characters** and must not be reused across systems. Employees must use the Company's approved password manager. Passwords must be changed immediately if compromise is suspected. Passwords must never be written down, shared in chat, or stored in code repositories.

5.4 **Joiner, mover, leaver.** Access is granted at joining based on role, changed when the person moves to a different role or team, and removed on separation. Movers must have any access that is no longer required removed within 5 working days.

5.5 **Access requests.** Access is requested through NexoraDesk. The request must state the business justification and duration. Requests must be approved by the reporting manager and, for Confidential or Restricted systems, by the data or system owner.

5.6 **Quarterly access reviews.** Managers must review the access of each direct report every quarter. IT & Security provides the list of entitlements, and managers must confirm or remove each item within 15 working days. Access that has not been used for 30 days is automatically suspended. Failure to complete reviews is reported to the Department Head and the CISO.

5.7 **Privileged accounts.** Administrative and privileged accounts are separate from ordinary user accounts, are held only by named individuals, and are subject to stricter controls in Section 6.

5.8 **Session controls.** Inactive sessions on corporate systems are locked or logged out after 15 minutes. Screen locks on laptops activate after 5 minutes of inactivity.

## 6. Production Access

6.1 **Principles.** Production access is the most sensitive form of access at Nexora. It is governed by the following principles, which apply to every employee, contractor, and intern regardless of grade:

- **No standing privileged access.** Write, deploy, and administrative access to production is granted only for a specific purpose, and expires automatically.
- **Least privilege.** The access level granted is the lowest level needed for the task.
- **Single control point.** All production access is requested, approved, granted, recorded, and revoked through **Nexora Access Vault (NAV)**. Direct logins to production servers, databases, and cloud consoles with personal or shared accounts are prohibited.
- **Strong authentication.** The user must be authenticated by Nexora SSO with MFA, must be connected to NexoraVPN (from any location other than a Nexora office network), and must use an approved, compliant Company device, as required by Sections 7 and 8 of this policy and Section 6.8 of NXR-HR-002.
- **Accountability.** Every action must be attributable to a named person and to a change ticket or incident record.

6.2 **Access levels.** Production access is divided into five levels:

| Level | Description | Eligible Persons | Approvers | Maximum Grant Duration |
|---|---|---|---|---|
| R0 | No production access | Default for all | — | — |
| R1 | Read-only access to dashboards and masked logs (no Restricted data) | All employees who have completed onboarding; contractors and interns with approval | Reporting manager | Standing for the period of employment, reviewed quarterly |
| R2 | Read access to unmasked logs or data that may include Confidential or Restricted data | Confirmed employees with a business need | Engineering Manager, data owner | 8 hours per grant |
| R3 | Write access: deployments, configuration changes, and database changes | Confirmed Engineering employees who have completed required training | Engineering Manager and an on-duty Security approver from IT & Security | 4 hours per grant |
| R4 | Privileged or administrative access (root, cloud administrator, database administrator) | Confirmed employees at grade L5 or above in SRE, Platform, or Security roles | VP of Engineering and CISO | 2 hours per grant |

6.3 **Employees on probation, contractors, and interns.**

(a) An employee who is on probation under Section 5 of NXR-HR-001 (that is, within the 6-month probation period, including any extension) may hold **R1 access only**. R1 access may be requested from the employee's 30th day of employment, after completion of the training in Section 13.

(b) Access above R1 for an employee on probation requires the **joint written approval of the VP of Engineering and the CISO**, is limited to the specific purpose stated, and must be recorded in the exceptions register (Section 14). In no circumstances may an employee on probation hold R4 access.

(c) Contractors, consultants, and interns are limited to R1 access. Access is granted for a maximum of **90 days** per grant. An extension requires the Department Head's approval and the CISO's approval. Contractors and interns are never granted R2, R3, or R4 access.

(d) On confirmation under Section 5.3 of NXR-HR-001, the employee becomes eligible to request R2 and R3 access in the normal way. Confirmation does not automatically grant any access.

6.4 **Just-in-time grants.** Access at levels R2, R3, and R4 is granted only through NAV for a specific task and duration, up to the maximum in Section 6.2. The request must state the reason and refer to a change ticket (for planned work) or an incident record (for incident work). Grants expire automatically at the end of the period. A user who needs more time must submit a new request.

6.5 **Break-glass access.** During a SEV1 or SEV2 incident (as defined in Section 10 of NXR-ENG-001), the Incident Commander may authorise emergency "break-glass" access to a named on-call engineer through NAV, without waiting for the standard approvals. Break-glass access is limited to 2 hours per grant and automatically notifies the CISO and the VP of Engineering. The Security team reviews each break-glass grant within one working day, and the Engineering Manager must document the justification in the incident record.

6.6 **Session recording and logging.** All R3 and R4 sessions are recorded by NAV, and commands are logged. Recordings and logs are retained for 12 months, in line with Section 11. Users must not attempt to disable, bypass, or tamper with recording.

6.7 **Change linkage.** Every R3 or R4 grant must be linked to an approved change under the change management process in Section 7 of NXR-ENG-001, or to an incident record. Grants without a valid ticket reference are automatically rejected.

6.8 **Quarterly production access review.** The CISO's team produces a production access report each quarter. Engineering Managers must certify within 15 working days that each person's access remains appropriate. Any R1 access not used within 90 days is removed.

6.9 **Location.** Production access is available only from India, from an approved device, and through NexoraVPN or a Nexora office network. Access from outside India is disabled by default. It may be enabled for a defined period only by the CISO, as part of an approval under Section 8 of NXR-HR-002.

6.10 **Prohibited actions.** The following are prohibited and constitute gross misconduct under Section 11.4 of NXR-HR-001:

- Accessing production systems without a valid NAV grant.
- Sharing credentials, tokens, or NAV sessions.
- Copying Restricted production data to personal or unapproved locations.
- Using production access for personal curiosity, including looking at customer or employee data without business need.
- Running commands not covered by the change or incident record.

## 7. Endpoint and Device Security

7.1 **Company devices.** Only devices issued and managed by Nexora IT may be used to access Confidential or Restricted information, or to access production. Devices must be enrolled in device management.

7.2 **Required controls.** Every Company device must have full-disk encryption enabled, endpoint detection and response (EDR) software running, an automatic screen lock of 5 minutes or less, an up-to-date operating system, and a host firewall enabled. Users must not disable or remove any of these controls.

7.3 **Patching.** Critical security patches must be applied within 7 days of release, high-severity patches within 14 days, and other patches within 30 days. Devices that are not compliant after the deadline lose access to corporate systems until they are updated.

7.4 **Personal devices.** Personal devices may be used only for email and calendar through the approved mobile application management profile. They must not be used to access source code, production, or Restricted data, or to store Company data.

7.5 **Removable media.** USB storage devices are disabled by default. Exceptions require the CISO's approval.

7.6 **Software installation.** Users must not install unapproved software. Requests for new software must go through NexoraDesk and are subject to Section 10.

7.7 **Lost or stolen devices.** A lost or stolen device must be reported to security@nexoratech.example and through NexoraDesk within **one hour** of discovery. IT & Security will remotely lock and, if necessary, wipe the device, and will review the potential exposure of data.

7.8 **Travel security.** Employees travelling with Company devices, whether on domestic or international business travel under NXR-FIN-001, must:

- Keep devices in hand luggage and never in checked baggage.
- Use the hotel room safe when leaving devices in a room.
- Connect to NexoraVPN on all hotel, airport, conference, and other public networks.
- Not use public USB charging points or borrowed cables.
- Avoid working on Restricted data in public places where screens can be seen.

For travel to countries that the CISO has designated as high risk, IT & Security will issue a loaner device with minimal data, and the employee must request this at least 5 working days before departure, as noted in Section 9.6 of NXR-FIN-001.

## 8. Network Security and Remote Access

8.1 **NexoraVPN.** Access to Internal, Confidential, and Restricted systems from outside a Nexora office network requires a connection to NexoraVPN. The VPN requires Nexora SSO with MFA and a compliant device. Split tunnelling is disabled. VPN sessions terminate after 12 hours and must be re-authenticated.

8.2 **Remote work.** Working remotely is permitted only under the approval process in Section 5 of NXR-HR-002. That policy requires reporting manager approval and a working VPN connection, and the controls in this section apply in addition.

8.3 **Office networks.** Office Wi-Fi is segmented into staff, guest, and device networks. Guests must use the guest network, which has no access to internal systems.

8.4 **Home networks.** Remote users must protect their home network as set out in Section 6.5 of NXR-HR-002.

8.5 **Network segmentation.** Production networks are separated from corporate and development networks. Access to production networks is only through controlled bastion services managed by NAV.

8.6 **Vulnerability management.** IT & Security scans internal and external systems weekly. Critical vulnerabilities must be remediated within 7 days, high within 30 days, and medium within 90 days. Engineering teams are responsible for fixing vulnerabilities in the systems they own, in line with Section 6 of NXR-ENG-001.

## 9. Security Incident Management

9.1 **What is an incident.** A security incident is any actual or suspected event that compromises the confidentiality, integrity, or availability of Company information or systems. Examples include lost or stolen devices, phishing that leads to disclosure of credentials, malware, unauthorised access, accidental disclosure of Restricted data, and any suspected breach of customer data.

9.2 **Reporting.** Every person who becomes aware of an actual or suspected security incident must report it **within one hour** by emailing security@nexoratech.example and raising a P1 or P2 ticket on NexoraDesk, or by calling the 24x7 security hotline published on NexoraHub. Reporting late or failing to report is a breach of this policy. No one will be penalised for reporting a mistake in good faith.

9.3 **Severity classification.** The SIRT classifies incidents into four priority levels. These correspond directly to the engineering incident severities in Section 10 of NXR-ENG-001:

| Security Priority | Engineering Severity | Description | Initial Response Target |
|---|---|---|---|
| P1 – Critical | SEV1 | Confirmed breach of Restricted data, or complete loss of a critical service | 15 minutes |
| P2 – High | SEV2 | Significant compromise or serious service degradation | 30 minutes |
| P3 – Medium | SEV3 | Limited impact; contained | 4 business hours |
| P4 – Low | SEV4 | Minor policy violation; no impact | 1 business day |

9.4 **Response process.** The SIRT, led by the CISO or a delegate, follows five stages: identification and triage, containment, eradication, recovery, and lessons learned. For incidents that also affect service availability, the SIRT works alongside the Engineering Incident Commander appointed under Section 10 of NXR-ENG-001.

9.5 **Regulatory notification.** The Head of Legal & Compliance and the DPO decide, with the CISO, whether an incident must be notified to regulators or individuals. Cyber incidents of the kinds specified by the Indian Computer Emergency Response Team (CERT-In) must be reported to CERT-In within 6 hours of detection. Personal data breaches must be notified to the Data Protection Board of India and affected individuals as required by the Digital Personal Data Protection Act, 2023. Nobody other than the Head of Legal & Compliance, the DPO, or the CISO may communicate with regulators or customers about a security incident.

9.6 **Post-incident review.** A written post-incident review must be completed within **5 business days** of closure for all P1 and P2 incidents. The review must identify root causes, corrective actions, owners, and dates, and is shared with the VP of Engineering and the Audit Committee where appropriate.

9.7 **Evidence.** Users must preserve evidence, including devices, emails, and logs, and must not delete or alter anything relating to a suspected incident.

## 10. Third Parties, Software, and Procurement

10.1 **Approved software and services.** Only software, cloud services, and applications on the approved list may be used with Company data. Requests for new tools must be submitted through NexoraDesk and are reviewed by IT & Security for security, privacy, and licensing risks.

10.2 **Purchases.** Purchases of software, subscriptions, and cloud services must follow the Procurement Policy. A purchase costing more than ₹25,000 also requires the approvals described in Section 4 of NXR-FIN-001, in addition to the IT & Security review. Employees must not use personal payment methods or Company cards to buy tools for use with Company data, as noted in Section 12.5 of NXR-FIN-001.

10.3 **Vendor risk assessment.** Vendors who will process Confidential or Restricted data must complete a security questionnaire and be assessed by IT & Security before contract signature. The contract must include confidentiality, data protection, breach notification within 24 hours, audit rights, and secure deletion of data on termination. The DPO reviews contracts involving personal data.

10.4 **Third-party access.** Third parties are given the minimum access required, for a fixed period, through NAV, and are subject to Section 6.3(c).

10.5 **Open-source and third-party code.** The use of open-source and third-party components in Nexora products is governed by Section 13 of NXR-ENG-001.

10.6 **Generative AI tools.** Employees must not enter Confidential or Restricted information into any generative AI tool that has not been approved by IT & Security. Approved tools are listed on NexoraHub.

## 11. Logging, Monitoring, and Retention

11.1 The Company logs and monitors access to corporate systems, VPN connections, device compliance, cloud consoles, and production. Logs include authentication events, privileged commands, configuration changes, and data access.

11.2 Security and audit logs are retained for **12 months**, of which the most recent 90 days are immediately searchable. Logs relating to a security incident or legal hold are retained for as long as needed.

11.3 Users should have no expectation of privacy when using Company systems or devices, to the extent permitted by law. Monitoring is carried out for security, compliance, and operational purposes only, and monitoring data is accessed only by authorised IT & Security staff.

11.4 Access to logs that contain personal data is limited and is itself logged. Monitoring must follow Section 9 of NXR-COC-001 and the DPDP Act, 2023.

## 12. Offboarding and Access Revocation

12.1 **Trigger.** People Operations informs IT & Security of resignations and terminations through NexoraPeople. For contractors and interns, access expires automatically on the end date.

12.2 **Timing.** Access is revoked as follows:

- **Voluntary separation:** all access, including Nexora SSO, NexoraVPN, email, source code repositories, and NAV grants, is disabled by **18:00 IST on the last working day**.
- **Termination for cause or gross misconduct:** all access is disabled immediately on notification, before the employee is informed if this is feasible.
- **Notice period:** where the reporting manager or Department Head believes there is a risk, access to Restricted systems and to production may be reduced during the notice period. Employees on notice must not be granted new R3 or R4 grants.

12.3 **Devices and materials.** Devices, tokens, and access cards must be returned as part of the clearance described in Section 10.4 of NXR-HR-001.

12.4 **Secrets and credentials.** Where the departing employee has held R3 or R4 access, or known shared secrets, the owning team must rotate the relevant secrets, keys, and tokens within 5 working days of separation, in line with Section 11 of NXR-ENG-001.

12.5 **Post-exit review.** IT & Security reviews the departing employee's access history for unusual downloads or activities in the 30 days before separation.

## 13. Security Awareness and Training

13.1 **Onboarding training.** Every new employee must complete Information Security Awareness training within **10 working days** of joining, as required by Section 4.5 of NXR-HR-001. Engineers must complete Secure Development Fundamentals before receiving merge rights, and production access training before requesting any access above R1.

13.2 **Annual refresher.** All employees must complete annual refresher training by the last working day of June. Employees who have not completed the training within 30 days after the due date have their access to non-essential systems suspended until it is complete.

13.3 **Phishing simulations.** IT & Security runs phishing simulations every quarter. Repeated failure results in additional coaching. Genuine phishing emails should be reported using the "Report Phish" button.

13.4 **Role-based training.** Engineers with R3 or R4 access, on-call engineers, and system administrators receive annual role-specific training.

## 14. Exceptions

14.1 Exceptions to this policy may be granted only by the CISO, and only when there is a documented business need and compensating controls. Exceptions that involve production access above R1 for employees on probation (Section 6.3) require the joint approval of the VP of Engineering and the CISO.

14.2 Every exception must be time-bound (a maximum of 90 days, renewable once), documented in the exceptions register, and reviewed by the CISO monthly.

14.3 An exception approved by the CHRO under Section 14 of NXR-HR-002 that relates to a technical control does not take effect until the CISO has also approved it.

## 15. Compliance and Consequences

15.1 The CISO monitors compliance with this policy and reports to the CEO and to the Audit Committee quarterly.

15.2 Breaches of this policy may result in loss of access and disciplinary action under Section 11 of NXR-HR-001, up to termination of employment or contract, and, where the law has been broken, referral to law enforcement.

15.3 Employees who report suspected violations in good faith are protected under Section 10 of NXR-COC-001.

## 16. Related Documents

- NXR-HR-001 Human Resources Policy Manual (Sections 4, 5, 10, 11, and 14)
- NXR-HR-002 Remote Work and Hybrid Working Policy (Sections 5, 6, 8, and 14)
- NXR-FIN-001 Travel and Expense Reimbursement Policy (Sections 4, 9, and 12)
- NXR-ENG-001 Engineering Guidelines and Operations Handbook (Sections 7, 8, 9, 10, 11, 12, and 13)
- NXR-COC-001 Code of Conduct and Ethics Policy (Sections 8, 9, and 10)

## 17. Document Control

| Version | Date | Description | Approved By |
|---|---|---|---|
| 1.0 | 1 April 2020 | Initial release | CEO |
| 3.0 | 1 April 2023 | Introduced Nexora Access Vault and just-in-time access | CEO |
| 4.0 | 1 April 2025 | Added access levels R0 to R4; aligned incident severities with engineering | CEO |
| 5.0 | 1 April 2026 | Added probation and contractor limits; added break-glass and generative AI rules | CEO |
