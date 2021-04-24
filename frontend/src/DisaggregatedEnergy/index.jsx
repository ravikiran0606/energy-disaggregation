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

const DisaggregatedEnergy = () => {
  const { Option } = Select;

  const [model, setModel] = useState('lstm');

  const [appliance, setAppliance] = useState('dishwasher');

  const [data, setData] = useState({
    isLoading: false,
    data: null,
    xAxis: null,
    yAxis: null,
    error: null,
  });

  const onChange = (info) => {
    if (info.file.status !== 'uploading') {
      setData({
        ...data,
        isLoading: true,
        data: null,
        xAxis: null,
        yAxis: null,
        error: null,
      });
      setAppliance('dishwasher');
    }
    if (info.file.status === 'done') {
      message.success(`${info.file.name} file uploaded successfully`);
      const { response } = info.file;
      const dishwasherData = response.dishwasher.map(({ timestamp, predictions}) => ({ x: moment(timestamp, 'YYYY-MM-DD HH:mm:ss').format('HH:mm:ss'), y: predictions }));
      const refrigeratorData = response.refrigerator.map(({ timestamp, predictions}) => ({ x: moment(timestamp, 'YYYY-MM-DD HH:mm:ss').format('HH:mm:ss'), y: predictions }));
      const mainsOneData = response.dishwasher.map(({ timestamp, mains_1 }) => ({ x: moment(timestamp, 'YYYY-MM-DD HH:mm:ss').format('HH:mm:ss'), y: mains_1 }));
      const mainsTwoData = response.dishwasher.map(({ timestamp, mains_2 }) => ({ x: moment(timestamp, 'YYYY-MM-DD HH:mm:ss').format('HH:mm:ss'), y: mains_2 }));
      const xAxis = {
        dishwasher: dishwasherData.map((d) => d.x),
        refrigerator: refrigeratorData.map((d) => d.x),
      };
      const yAxis = {
        dishwasher: [
          _min([_min(dishwasherData, (d) => d.y).y, _min(mainsOneData, (d) => d.y).y, _min(mainsTwoData, (d) => d.y).y]),
          _max([_max(dishwasherData, (d) => d.y).y, _max(mainsOneData, (d) => d.y).y, _max(mainsTwoData, (d) => d.y).y]),
        ],
        refrigerator: [
          _min([_min(refrigeratorData, (d) => d.y).y, _min(mainsOneData, (d) => d.y).y, _min(mainsTwoData, (d) => d.y).y]),
          _max([_max(refrigeratorData, (d) => d.y).y, _max(mainsOneData, (d) => d.y).y, _max(mainsTwoData, (d) => d.y).y]),
        ],
      };
      setData({
        ...data,
        isLoading: false,
        data: {
          dishwasher: dishwasherData,
          refrigerator: refrigeratorData,
        },
        xAxis,
        yAxis,
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
          <Select defaultValue="lstm" style={{ width: 120 }} onChange={setModel}>
            <Option value="lstm">LSTM</Option>
            <Option value="cnn">CNN</Option>
          </Select>
        </div>
        <div className="form-field">
          <UploadFile onChange={onChange} model={model} />
        </div>
        {
          renderIf(() => data.data && data.data.dishwasher && data.data.refrigerator, () => (
            <div className="form-field">
              <Select defaultValue="dishwasher" style={{ width: 150 }} onChange={setAppliance}>
                <Option value="dishwasher">Dishwasher</Option>
                <Option value="refrigerator">Refrigerator</Option>
              </Select>
            </div>
          ))
        }
      </div>
      {
        renderIf(() => data.data && appliance && data.data[appliance], () => (
          <div className="data-wrapper">
            <Chart values={data.data[appliance]} xAxis={data.xAxis[appliance]} yAxis={data.yAxis[appliance]} />
            <DataTable data={data.data[appliance]} />
          </div>
        ))
      }
    </div>
  )
};

export default DisaggregatedEnergy;
