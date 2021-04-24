import React from 'react';
import { Upload, Button } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

const UploadFile = ({ onChange, model }) => {
  const uploadProps = {
    name: 'file',
    action: `http://localhost:5600/disaggregate?model=${model}`,
    onChange: onChange,
    maxCount: 1,
  };

  return (
    <Upload {...uploadProps}>
      <Button icon={<UploadOutlined />}>Click to Upload</Button>
    </Upload>
  )
};

export default UploadFile;
