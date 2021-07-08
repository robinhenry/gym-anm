# Contribution Guidelines
First off, thanks for taking the time to contribute! 🎉

## Code of Conduct
Help us keep this project open and inclusive. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). 

## Report bugs using Github's [issues](https://github.com/robinhenry/gym-anm/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/robinhenry/gym-anm/issues).

## Write bug reports with detail, background, and sample code
Great Bug Reports tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)
- If you know who should look into this, then assign the issue. If you don't, just assign it to @robinhenry.

## Propose changes to the codebase
Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://guides.github.com/introduction/flow/index.html)). We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Request @robinhenry as the reviewer.
7. Issue that pull request!

## Propose a novel `gym-anm` environment
If you designed a new `gym-anm` environment and would like it to be added to the project, you can do so as follows:

- On your own fork of the repository, create a new folder in [gym_anm/envs/](gym_anm/envs) and name it after the environment you designed.
- Make sure all your code files are contained within that repository.
- Write a documentation page for your environment in [docs/source/topics/](docs/source/topics) and add it to the documenation index [docs/source/index.rst](docs/source/index.rst).
- Register your environment with `Gym` in [gym_anm/\_\_init\_\_.py](gym_anm/__init__.py).
- Propose your environment to be added to the codebase using a Pull Request (see above).

## Coding Conventions
Please follow the same coding conventions as used in the existing codebase. 

### Thank you for your contribution! ❤️

