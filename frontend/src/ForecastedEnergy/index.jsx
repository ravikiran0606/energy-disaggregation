import React, { useState } from 'react';
import { message, Select } from 'antd';
import _min from 'lodash/min';
import _max from 'lodash/max';
import moment from 'moment';
import UploadFile from './FileUpload';

import './index.css';
import Chart from './Chart';
import { renderIf } from '../utils';
import DataTable from './DataTable';

const ForecastedEnergy = () => {
  const [data, setData] = useState({
    isLoading: false,
    data: null,
    xAxis: null,
    error: null,
  });

  const onChange = (info) => {
    if (info.file.status !== 'uploading') {
      setData({
        ...data,
        isLoading: true,
        data: null,
        xAxis: null,
        error: null,
      });
    }
    if (info.file.status === 'done') {
      message.success(`${info.file.name} file uploaded successfully`);
      const { response } = info.file;
      const historicalData = [];
      const forecastedData = [];
      const xAxis = [];
      response.data.forEach(({ timestamp, flag, output}) => {
        if (flag === 0) {
          historicalData.push({ x: moment(timestamp, 'YYYY-MM-DD HH:mm:ss').format('MM/DD HH:mm'), y: output });
        }
        if (flag === 1) {
          forecastedData.push({ x: moment(timestamp, 'YYYY-MM-DD HH:mm:ss').format('MM/DD HH:mm'), y: output });
        }
        xAxis.push(moment(timestamp, 'YYYY-MM-DD HH:mm:ss').format('MM/DD HH:mm'));
      });
      historicalData.push(forecastedData[0]);
      setData({
        ...data,
        isLoading: false,
        data: {
          historical: historicalData,
          forecast: forecastedData,
        },
        xAxis,
      });
    } else if (info.file.status === 'error') {
      message.error(`${info.file.name} file upload failed.`);
      setData({
        ...data,
        isLoading: false,
        error: true,
      });
    }
  };

  return (
    <div className="disaggregated-container">
      <div className="form-wrapper">
        <div className="form-field">
          <UploadFile onChange={onChange} />
        </div>
      </div>
      {
        renderIf(() => data.data && data.data.historical && data.data.forecast, () => (
          <div className="data-wrapper">
            <Chart
              historical={data.data.historical}
              forecast={data.data.forecast}
              xAxis={data.xAxis}
            />
            <DataTable historical={data.data.historical} forecast={data.data.forecast} />
          </div>
        ))
      }
    </div>
  )
};

export default ForecastedEnergy;
