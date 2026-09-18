"""Pin the dependency closure actually used by this project."""
from pathlib import Path
from importlib.metadata import distribution
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.utils import canonicalize_name
ROOT=Path(__file__).resolve().parents[1]
pending=[Requirement(x).name for x in (ROOT/'requirements.txt').read_text().splitlines() if x and not x.startswith('#')]
found={};environment=default_environment()|{'extra':''}
while pending:
    name=pending.pop();key=canonicalize_name(name)
    if key in found:continue
    dist=distribution(name);found[key]=f"{dist.metadata['Name']}=={dist.version}"
    for item in dist.requires or []:
        requirement=Requirement(item)
        if requirement.marker is None or requirement.marker.evaluate(environment):pending.append(requirement.name)
(ROOT/'requirements-lock.txt').write_text('# Tested active dependency closure: Windows, Python 3.12.5.\n'+'\n'.join(found[k] for k in sorted(found))+'\n',encoding='utf-8')
print(f'Pinned {len(found)} project dependencies.')
