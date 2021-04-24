import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Skeleton, Result } from 'antd';
import { FrownOutlined } from '@ant-design/icons';
import FilterForm from './FilterForm';
import moment from 'moment';
import Chart from './Chart';
import DataTable from './DataTable';

import './index.css';

const MainDisaggregation = () => {
  const [formValues, setFormValues] = useState({
    house: '1',
    mains: '1',
    date: null,
    time: null,
    endTime: null,
    maxDate: null,
    minDate: null,
  });

  const [data, setData] = useState({
    isLoading: false,
    data: null,
    xAxis: null,
    error: null,
  });

  const handleChange = (key, value) => {
    const newValues = {};
    if (key === 'house') {
      newValues.maxDate = null;
      newValues.minDate = null;
      newValues.date = null;
    }
    if (key === 'time') {
      newValues.endTime = moment(value, 'HH:mm:ss').add(5, 'm').format('HH:mm:ss');
    }
    setFormValues({
      ...formValues,
      ...newValues,
      [key]: value,
    });
    setData({
      ...data,
      data: null,
      xAxis: null,
      error: null,
    })
  };

  useEffect(() => {
    if (formValues.house && formValues.maxDate === null && formValues.minDate === null) {
      axios.get(`http://localhost:5600/house/${formValues.house}`)
        .then(({ data }) => {
          setFormValues({
            ...formValues,
            maxDate: data.maxDate,
            minDate: data.minDate,
            date: data.minDate,
            time: '14:00:00',
            endTime: '14:05:00',
          });
        });
    }
  });

  useEffect(() => {
    const { house, mains, date, time, endTime } = formValues;
    if (!data.isLoading && data.data === null && data.error === null && house && mains && date && time && endTime) {
      setData({
        ...data,
        isLoading: true,
        data: null,
        xAxis: null,
        error: null,
      });
      const start = `${date} ${time}`;
      const end = `${date} ${endTime}`;
      axios.get(`http://localhost:5600/data/${house}/${mains}?start=${start}&end=${end}`)
        .then(({ data: responseData }) => {
          if (Object.keys(responseData).length === 0) {
            setData({
              ...data,
              isLoading: false,
              error: 'no data',
            });
          } else {
            const keys = Object.keys(responseData);
            let mainsAttr = 'mains_1'
            if (mains === '2') {
              mainsAttr = 'mains_2'
            }
            const modifiedData = keys.map((key) => ({
              x: moment(key, 'YYYY-MM-DD HH:mm:ss').format('HH:mm:ss'),
              y: responseData[key][mainsAttr]
            }));
            setData({
              ...data,
              isLoading: false,
              data: modifiedData,
              xAxis: keys.map((key) => moment(key, 'YYYY-MM-DD HH:mm:ss').format('HH:mm:ss')),
            });
          }
        })
        .catch(() => {
          setData({
            ...data,
            isLoading: false,
            error: 'server error',
          });
        });
    } 
  });

  return (
    <div className="mains-container">
      <div>
        <h3>Total Energy Consumption</h3>
        <FilterForm formValues={formValues} handleChange={handleChange} />
      </div>
      {
        data.isLoading &&
        <Skeleton active />
      }
      {
        data.data !== null && data.data.length > 0 && data.xAxis !== null &&
        <div className="mains-wrapper">
          <Chart values={data.data} xAxis={data.xAxis} />
          <DataTable data={data.data} />
        </div>
      }
      {
        data.error === 'no data' &&
        <Result
          icon={<FrownOutlined />}
          title="No Data"
          subTitle="No data was found for the given parameters"
        />
      }
      {
        data.error === 'server error' &&
        <Result
          status="500"
          title="500"
          subTitle="Sorry, something went wrong."
        />
      }
    </div>
  );
};

export default MainDisaggregation;
