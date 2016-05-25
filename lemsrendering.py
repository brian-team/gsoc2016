from brian2.parsing.rendering import NodeRenderer

class LEMSRenderer(NodeRenderer):
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update({
          # BinOp
          'Pow': '^',
          # Unary ops
          # Bool ops
          })
    
    def render_NameConstant(self, node):
        # In Python 3.4, None, True and False go here
        return {True: 'true',
                False: 'false'}.get(node.value, node.value)
