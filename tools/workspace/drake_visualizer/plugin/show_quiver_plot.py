# Note that this script runs in the main context of drake-visualizer,
# where many modules and variables already exist in the global scope.

from director import lcmUtils
from director import applogic
from director import objectmodel as om
from director import visualization as vis
from director.debugVis import DebugData
import numpy as np
from PythonQt import QtCore, QtGui

import drake as lcmdrakemsg

from drake.tools.workspace.drake_visualizer.plugin import scoped_singleton_func
from sortedcontainers import SortedDict

# TODO(seancurtis-tri) Refactor this out of show_hydroelastic_contact.py and
#                      show_point_pair_contact.py.
class ArrowVisModes:
    '''Common specification of contact visualization modes'''
    @staticmethod
    def get_mode_string(mode):
        if mode == ArrowVisModes.kFixedLength:
            return "Fixed Length"
        elif mode == ArrowVisModes.kGlobalScale:
            return "Scaled"
        elif mode == ArrowVisModes.kDirectionLength:
            return "Auto-scale"
        else:
            return "Unrecognized mode"

    @staticmethod
    def get_modes():
        return (ArrowVisModes.kFixedLength, ArrowVisModes.kGlobalScale,
                ArrowVisModes.kDirectionLength)

    @staticmethod
    def get_mode_docstring(mode):
        if mode == ArrowVisModes.kFixedLength:
            return "Arrows have fixed length equal to global scale"
        elif mode == ArrowVisModes.kGlobalScale:
            return "Arrow scale scaled by global scale"
        elif mode == ArrowVisModes.kDirectionLength:
            return "Arrow scale set by direction vector."
        else:
            return "unrecognized mode"

    kFixedLength = 0
    kGlobalScale = 1
    kDirectionLength = 2


class _ContactConfigDialog(QtGui.QDialog):
    '''A simple dialog for configuring the contact visualization'''
    def __init__(self, visualizer, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.setWindowTitle("Arrow Visualization")
        layout = QtGui.QGridLayout()
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)

        row = 0

        # Magnitude representation
        layout.addWidget(QtGui.QLabel("Arrow Length"), row, 0)
        self.magnitude_mode = QtGui.QComboBox()
        modes = ArrowVisModes.get_modes()
        mode_labels = [ArrowVisModes.get_mode_string(m) for m in modes]
        self.magnitude_mode.addItems(mode_labels)
        self.magnitude_mode.setCurrentIndex(visualizer.magnitude_mode)
        mode_tool_tip = 'Determines how each arrow in the quiver plot is visualized:\n'
        for m in modes:
            mode_tool_tip += '  - {}: {}\n'.format(
                ArrowVisModes.get_mode_string(m),
                ArrowVisModes.get_mode_docstring(m))
        self.magnitude_mode.setToolTip(mode_tool_tip)
        layout.addWidget(self.magnitude_mode, row, 1)
        row += 1

        # Global scale.
        layout.addWidget(QtGui.QLabel("Global scale"), row, 0)
        self.global_scale = QtGui.QLineEdit()
        self.global_scale.setToolTip(
            'All visualized forces are multiplied by this scale factor (must '
            'be non-negative)')
        validator = QtGui.QDoubleValidator(0, 100, 3, self.global_scale)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.global_scale.setValidator(validator)
        self.global_scale.setText("{:.3f}".format(visualizer.global_scale))
        layout.addWidget(self.global_scale, row, 1)
        row += 1

        # Magnitude cut-off.
        layout.addWidget(QtGui.QLabel("Minimum force"), row, 0)
        self.min_magnitude = QtGui.QLineEdit()
        self.min_magnitude.setToolTip('Arrows with a length less than this '
                                      'value will not be visualized (must be '
                                      '> 1e-10)')
        validator = QtGui.QDoubleValidator(1e-10, 100, 10, self.min_magnitude)
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.min_magnitude.setValidator(validator)
        self.min_magnitude.setText("{:.3g}".format(visualizer.min_magnitude))
        layout.addWidget(self.min_magnitude, row, 1)
        row += 1

        # Accept/cancel.
        btns = QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel
        buttons = QtGui.QDialogButtonBox(btns, QtCore.Qt.Horizontal, self)
        buttons.connect('accepted()', self.accept)
        buttons.connect('rejected()', self.reject)
        layout.addWidget(buttons, row, 0, 1, 2)

        self.setLayout(layout)


# TODO(SeanCurtis): This would be better extracted out of *this* plugin
def get_sub_menu_or_make(menu, menu_name):
    for a in menu.actions():
        if a.text == menu_name:
            return a.menu()
    return menu.addMenu(menu_name)


class ContactVisualizer(object):
    def __init__(self):
        self._folder_name = 'Point Pair Contact Results'
        self._name = "Contact Visualizer"
        self._enabled = False
        self._sub = None

        # Visualization parameters
        # TODO(SeanCurtis-TRI): Find some way to persist these settings across
        #  invocations of drake visualizer. Config file, environment settings,
        #  something.
        self.magnitude_mode = ArrowVisModes.kFixedLength
        self.global_scale = 0.3
        self.min_magnitude = 1e-4

        # A sorted map or arrows with their expiry time stamp for keys.
        self.arrows_to_display = SortedDict()

        menu_bar = applogic.getMainWindow().menuBar()
        plugin_menu = get_sub_menu_or_make(menu_bar, '&Plugins')
        contact_menu = get_sub_menu_or_make(plugin_menu, '&Contacts')
        self.configure_action = contact_menu.addAction(
            "&Configure Force Vector for Point Contacts")
        self.configure_action.connect('triggered()', self.configure_via_dialog)

        self.set_enabled(True)
        self.update_screen_text()

    def configure_via_dialog(self):
        '''Configures the visualization'''
        dlg = _ContactConfigDialog(self)
        if dlg.exec_() == QtGui.QDialog.Accepted:
            # TODO(SeanCurtis-TRI): Cause this to redraw any forces that are
            #  currently visualized.
            self.magnitude_mode = dlg.magnitude_mode.currentIndex
            self.global_scale = float(dlg.global_scale.text)
            self.min_magnitude = float(dlg.min_magnitude.text)
            self.update_screen_text()

    def update_screen_text(self):
        folder = om.getOrCreateContainer(self._folder_name)
        my_text = 'Contact vector: {}'.format(
            ArrowVisModes.get_mode_string(self.magnitude_mode))

        # TODO(SeanCurtis-TRI): Figure out how to anchor this in the bottom-
        #  right corner as opposed to floating in the middle.
        w = applogic.getCurrentRenderView().size.width()
        vis.updateText(my_text, 'contact_text',
                       **{'position': (w/2, 10), 'parent': folder})

    def add_subscriber(self):
        if self._sub is not None:
            return

        self._sub = lcmUtils.addSubscriber(
            'QUIVER_PLOT',
            messageClass=lcmdrakemsg.lcmt_quiver_plot,
            callback=self.handle_message)
        print(self._name + " subscriber added.")

    def remove_subscriber(self):
        if self._sub is None:
            return

        lcmUtils.removeSubscriber(self._sub)
        self._sub = None
        om.removeFromObjectModel(om.findObjectByName(self._folder_name))
        print(self._name + " subscriber removed.")

    def is_enabled(self):
        return self._enabled

    def set_enabled(self, enable):
        self._enabled = enable
        if enable:
            self.add_subscriber()
            self.configure_action.setEnabled(True)
        else:
            self.remove_subscriber()
            self.configure_action.setEnabled(False)
            # Removes the folder completely.
            om.removeFromObjectModel(om.findObjectByName(self._folder_name))


    def handle_message(self, msg):
        # Limits the rate of message handling, since redrawing is done in the
        # message handler.
        self._sub.setSpeedLimit(1000)

        # Removes the folder completely.
        om.removeFromObjectModel(om.findObjectByName(self._folder_name))

        # Recreates folder.
        folder = om.getOrCreateContainer(self._folder_name)

        # Clean up expired arrows.
        for viz_timeout in self.arrows_to_display.keys():
          if viz_timeout < msg.timestamp:
            self.arrows_to_display.pop(viz_timeout)
          else: break

        for arrow in msg.arrows:
            point = np.array([arrow.point[0],
                              arrow.point[1],
                              arrow.point[2]])
            direction = np.array([arrow.direction[0],
                              arrow.direction[1],
                              arrow.direction[2]])
            duration = arrow.duration
            mag = np.linalg.norm(direction)

            if mag < self.min_magnitude:
              continue

            if duration < 1e3:
              continue

            if duration > 1e6:
              duration = 1e6

            viz_timeout = msg.timestamp + duration
            self.arrows_to_display[viz_timeout] = [(point, direction)]

        for arrows in self.arrows_to_display.values():
          d = DebugData()
          for p, v in arrows:
            d.addArrow(start=p,
                       end=p + v,
                       tubeRadius=0.005,
                       headRadius=0.01)
            vis.showPolyData(d.getPolyData(), str(p), parent=folder,
                             color=[0.2, 0.8, 0.2])
        
        self.update_screen_text()


@scoped_singleton_func
def init_visualizer():
    # Create a visualizer instance.
    my_visualizer = ContactVisualizer()
    # Adds to the "Tools" menu.
    applogic.MenuActionToggleHelper(
        'Tools', my_visualizer._name,
        my_visualizer.is_enabled, my_visualizer.set_enabled)
    return my_visualizer


# Activate the plugin if this script is run directly; store the results to keep
# the plugin objects in scope.
if __name__ == "__main__":
    contact_viz = init_visualizer()
