import React from 'react';
import { Table } from 'antd';

const DataTable = ({ data, mainsOne, mainsTwo }) => {
  const mergedData = data.map((val, index) => ({
    timestamp: val.x,
    prediction: val.y,
    mains1: mainsOne[index].y,
    mains2: mainsTwo[index].y,
  }));
  
  const columns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
    },
    {
      title: 'Mains 1',
      dataIndex: 'mains1',
    },
    {
      title: 'Mains 2',
      dataIndex: 'mains2',
    },
    {
      title: 'Energy Consumed',
      dataIndex: 'prediction',
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={mergedData}
      pagination={{
        position: ['topLeft', 'none'],
        showSizeChanger: false,
      }}
    />
  );
};

export default DataTable;
