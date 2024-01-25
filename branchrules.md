## Some rules not to get confused with branches
#### Quite temporary, we'll see what works for us
 - **main** and **dev** exist continuously and are always expected to run without errors
 - Now we are doing something like a standard "Git-Flow" strategy (https://www.gitkraken.com/learn/git/git-flow#the-git-flow-diagram). Later, when the code changes less fast, we might want to get rid of **dev**.
 - **main** contains the latest version we want users to run. **dev** contains breaking changes, belonging to a next release.
 - Every single new feature or fix belongs to a separate, dedicated temporary branch. Conceptually different things shouldn't share one temporary branch.
     - If this new thing is intended for **dev** only, the respective branch should be based on the most recent state of **dev**, and in the end should be merged into **dev** (and then be deleted). These are called 'feature' branches in git-flow terminology. 
     - Otherwise, it should be based on most recent state of **main** and finally be merged into **main** (and if relevant there too, also to **dev**) and then be deleted. These are called 'hotfix' branches in git-flow terminology. 
     - These temporary branches are allowed to fail while development, it is more important to continuously push stuff online than making sure everything online is perfect.
 - Updating main:
     - When making changes to main, we update the version number in Project.toml following Semantic Versioning (https://semver.org/) and then create a corresponding tag in GitHub. (I think the latter step can be automated once we register the package.) 
     - Before breaking changes, we might want to create an archive branch from the pre-update version of main, so that we can support older versions if necessary.
     - It is ok to merge some changes together into a 'release' branch before merging to **main**. (So that we don't change the version number, and we don't require reviews from each other  too often) It is still important that separate changes have their own branches first, since this makes it easier to understand the final pull requests into **main** than just a linear series of unrelated commits would be. After merging, release branches are to be deleted.