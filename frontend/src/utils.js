export const renderIf = (condition, component) => (condition()) ? component() : null;
