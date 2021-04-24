import React from 'react';
import { Select, DatePicker, TimePicker, Space } from 'antd';
import moment from 'moment';

import './FilterForm.css';

const FilterForm = ({ formValues, handleChange }) => {
  const { Option } = Select;

  return (
    <div className="filter-form">
      <Select className="form-field" defaultValue="1" style={{ width: 120 }} onChange={(value) => handleChange('house', value)}>
        <Option value="1">House 1</Option>
        <Option value="2">House 2</Option>
        <Option value="3">House 3</Option>
        <Option value="4">House 4</Option>
        <Option value="5">House 5</Option>
        <Option value="6">House 6</Option>
      </Select>
      <Select className="form-field" defaultValue="1" style={{ width: 120 }} onChange={(value) => handleChange('mains', value)}>
        <Option value="1">Mains 1</Option>
        <Option value="2">Mains 2</Option>
      </Select>
      {formValues.maxDate && formValues.minDate &&
        <Space className="form-field" direction="vertical">
          <DatePicker
            format="YYYY-MM-DD"
            disabledDate={(current) => current && current < moment(formValues.minDate) && current > moment(formValues.maxDate)}
            defaultValue={moment(formValues.minDate)}
            onChange={(_, value) => handleChange('date', value)}
          />
        </Space>
      }
      {formValues.date &&
        <TimePicker
          className="form-field" 
          format="HH:mm:ss"
          defaultValue={moment(formValues.time, 'HH:mm:ss')}
          onChange={(_, value) => handleChange('time', value)}
          minuteStep={5}
        />
      }
    </div>
  )
};

export default FilterForm;
