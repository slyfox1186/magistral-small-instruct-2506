# Discovery Questions

These questions will help understand the scope and priorities for identifying and fixing incorrect logic in the codebase.

## Q1: Should I prioritize fixing security vulnerabilities and potential data exposure risks first?
**Default if unknown:** Yes (security issues can have immediate impact on users and should be addressed with highest priority)

## Q2: Do you want me to fix performance bottlenecks and memory leak issues that could affect system stability?
**Default if unknown:** Yes (stability issues directly impact user experience and system reliability)

## Q3: Should I include fixing code quality issues like commented-out code, magic numbers, and poor error handling?
**Default if unknown:** Yes (clean code improves maintainability and reduces future bugs)

## Q4: Do you want me to address architectural issues like tight coupling and missing modularization?
**Default if unknown:** No (architectural refactoring is time-consuming and might break existing functionality)

## Q5: Should I fix configuration and deployment issues like hardcoded values and missing test infrastructure?
**Default if unknown:** Yes (proper configuration management prevents environment-specific bugs)