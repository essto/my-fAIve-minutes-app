const React = require('react');
const { View } = require('react-native');

const BlurView = ({ children, ...props }) => {
  return React.createElement(View, { ...props, testID: 'blur-view' }, children);
};

module.exports = {
  BlurView,
};
