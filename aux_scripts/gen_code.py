#
## generates code from the .pyx_tmpl and .pxd_tmpl files
#
from jinja2 import Template, Environment,FileSystemLoader

wanted_types=[('I','int','int32'),
              ('F','float','float32'),
              ('D','double','float64'),
              ('V','void','FIXME')]

env=Environment(loader=FileSystemLoader('.'))
env.globals['wanted_types']=wanted_types

for basename in ['pyx_src/lurrn/sparsmat.pxd',
                 'pyx_src/lurrn/sparsmat.pyx']:
    tmpl=env.get_template(basename+'_tmpl')
    f=file(basename,'w')
    f.write(tmpl.render())
    f.close()

