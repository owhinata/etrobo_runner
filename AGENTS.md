# Repository Guidelines

This repository provides a scan line follower node for **ET Robocon**.

## Environment
- Ubuntu 22.04  
- ROS 2 Humble  
- Gazebo 11.10.2  
- Python 3.10  

For simulation:  
- [etrobo_simulator](https://github.com/owhinata/etrobo_simulator)

## Development
- Language: Python and C++  
- Follow the PEP 8 coding style.  
- Follow the Google C++ Style Guide.
- Run the code formatter before committing (C++: `clang-format` using the repo `.clang-format`).
- When adding new features, make only the minimum necessary changes.  
  Refactoring (e.g., variable renaming or other code improvements) is **not** allowed unless explicitly requested.
- The pull-request description **must include** the following sections:
  - **Requirement** – copy the task details **verbatim**
  - **Change Summary**
  - **Testing**

## Commit Messages
- Language: English.  
- Style: Prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`).  
- Title line: one-line summary, 80 characters or fewer.  
- Body: optional; wrap lines at 80 characters; explain what and why.  
- Tone: imperative mood (e.g., "add", "fix", "update").  
- Scope: optional short scope in parentheses (e.g., `feat(ui): ...`).  
- Formatting: no trailing period in title; separate title/body with a blank line.  

## Documentation
- Write all documentation, code comments, and branch names in **English**.  
- If the design policy changes, **update `doc/DESIGN.md` as well**.

## Testing
Run the following after making changes:
