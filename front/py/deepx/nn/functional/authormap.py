defaultauthor=dict({
        #io
        'print':'miaobyte',
        #init
        'uniform':'miaobyte',
        'constant':'miaobyte',
        'arange':'miaobyte',
        'normal':'miaobyte',
        #elementwise
        'add':'miaobyte',
        'addscalar':'miaobyte',
        'sub':'miaobyte',
        'subscalar':'miaobyte',
        'mul':'miaobyte',
        'mulscalar':'miaobyte',
        'div':'miaobyte',
        'divscalar':'miaobyte',
        'rdiv':'miaobyte',
        'rdivscalar':'miaobyte',
        #
        'invert':'miaobyte',
        #
        'min':'miaobyte',
        'minscalar':'miaobyte',
        'max':'miaobyte',
        'maxscalar':'miaobyte',
        #
        'less': 'miaobyte',
        'greater': 'miaobyte',
        'equal': 'miaobyte',
        'notequal': 'miaobyte',
        #
        'exp':'miaobyte',
        'log':'miaobyte',
        'pow':'miaobyte',
        'powscalar':'miaobyte',
        'rpowscalar':'miaobyte',
        'sqrt':'miaobyte',
        #
        'dropout':'miaobyte',
        #changeshape
        'reshape':'miaobyte',
        'transpose':'miaobyte',
        'broadcastTo':'miaobyte',
        'concat':'miaobyte',
        'indexselect':'miaobyte',
        #matmul
        # 'matmul':'miaobyte',
        'matmul':'cublas',
        #reduce
        'sum':'miaobyte',
        'prod':'miaobyte',
        'reducemax':'miaobyte',
        'reducemin':'miaobyte'     
    })