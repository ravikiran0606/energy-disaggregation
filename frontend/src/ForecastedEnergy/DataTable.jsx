import React from 'react';
import { Table } from 'antd';

const DataTable = ({ historical, forecast }) => {
  const mergedData = [];

  historical.forEach((val) => {
    mergedData.push({
      timestamp: val.x,
      value: val.y,
      forecast: 'No',
    });
  });

  forecast.forEach((val) => {
    mergedData.push({
      timestamp: val.x,
      value: val.y,
      forecast: 'Yes',
    });
  });
  
  const columns = [
    {
      title: 'Time',
      dataIndex: 'timestamp',
    },
    {
      title: 'Energy Consumed',
      dataIndex: 'value',
    },
    {
      title: 'Predicted',
      dataIndex: 'forecast',
    }
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
