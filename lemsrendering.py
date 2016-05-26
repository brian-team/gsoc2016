from brian2.parsing.rendering import NodeRenderer

class LEMSRenderer(NodeRenderer):
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update({
          # BinOp
          'Pow': '^',
          # ??? 'Mod': '%', 
          # Compare
          'Lt': '.lt.',
          'LtE': '.le.',
          'Gt': '.gt.',
          'GtE': '.ge.',
          'Eq': '.eq.',
          'NotEq': '.ne.',
          # Unary ops
          'Not': '.not.',
          # Bool ops
          'And': '.and.',
          'Or': '.or.'
          })
