defaultauthor=dict({
        #io
        'print':'miaobyte',
        #init
        'uniform':'miaobyte',
        'constant':'miaobyte',
        'arange':'miaobyte',
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

        'compare':'miaobyte',
        'min':'miaobyte',
        'minscalar':'miaobyte',
        'max':'miaobyte',
        'maxscalar':'miaobyte',
        'exp':'miaobyte',
        'log':'miaobyte',
        'pow':'miaobyte',
        'powscalar':'miaobyte',
        'sqrt':'miaobyte',
        #changeshape
        'reshape':'miaobyte',
        'transpose':'miaobyte',
        'broadcastTo':'miaobyte',
        'concat':'miaobyte',
        #matmul
        # 'matmul':'miaobyte',
        'matmul':'cublas',
        #reduce
        'sum':'miaobyte',
        'prod':'miaobyte',
        'reducemax':'miaobyte',
        'reducemin':'miaobyte'     
    })