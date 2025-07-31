
---

### `SECURITY.md`

```markdown
# Security Policy

We take security seriously and want to ensure that our repository and
its users are protected. This document outlines how to report
vulnerabilities and the security practices we follow.

## Reporting a vulnerability

If you discover a security vulnerability or find something that
appears unsafe (for example, leaked credentials or malicious
dependencies), **do not create a public issue or pull request**. Instead,
please email the maintainers at **security@example.com** with a detailed
description of the problem and steps to reproduce it. If possible,
include the commit hash or notebook name where the issue occurs.

We will acknowledge receipt of your report within two business days
and work with you to understand and resolve the issue as quickly as
possible. When the vulnerability is fixed, we will credit you (with
permission) in the release notes.

## Security best practices we follow

GitHub offers several built‑in security features to help maintain a
safe project. We use and recommend the following tools:

* **Dependency monitoring (Dependabot).** We enable Dependabot to
  automatically scan for outdated or vulnerable dependencies and
  suggest updates. Keeping dependencies up to date reduces exposure
  to known vulnerabilities.
* **Secret scanning and push protection.** Secret scanning detects
  API keys and tokens checked into the repository and alerts
  maintainers so they can be removed. Push protection can block
  accidental commits containing secrets before they are merged.
* **Code scanning.** We use GitHub’s CodeQL code scanning to
  identify potential security issues in our code. Alerts are
  triaged and addressed before releases.

## Keeping your contributions secure

When contributing to this project, please follow these guidelines:

* **Never commit secrets.** Do not include API keys, passwords or
  personal data in commits, notebooks or issues. If you accidentally
  commit a secret, remove it immediately, rotate the key, and alert
  the maintainers via email.
* **Validate inputs.** If your contribution involves code that takes
  user input, ensure that inputs are validated and sanitised to
  prevent injection attacks.
* **Use supported dependencies.** Only add dependencies from
  reputable sources. Pin dependency versions where possible to
  minimise supply‑chain risk.
* **Keep dependencies updated.** Regularly update your environment
  with `pip install -U` or similar commands to fetch the latest
  security patches.

By following these practices, we can work together to build a
trustworthy and secure resource for the AI/ML community.
