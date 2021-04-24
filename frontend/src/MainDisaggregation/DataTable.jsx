import React from 'react';
import { Table } from 'antd';

const DataTable = ({ data }) => {
  const columns = [
    {
      title: 'Time',
      dataIndex: 'x',
    },
    {
      title: 'Energy Consumed',
      dataIndex: 'y',
    },
  ];

  console.log(data);

  return (
    <Table
      columns={columns}
      dataSource={data}
      pagination={{
        position: ['topLeft', 'none'],
        showSizeChanger: false,
      }}
    />
  );
};

export default DataTable;
