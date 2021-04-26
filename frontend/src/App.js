import React, { useState } from 'react';
import { Layout, Menu } from 'antd';

import 'antd/dist/antd.css';
import MainDisaggregation from './MainDisaggregation';
import DisaggregatedEnergy from './DisaggregatedEnergy';
import ForecastedEnergy from './ForecastedEnergy';
import { renderIf } from './utils';

const App = () => {
  const { Header, Content } = Layout;
  const [option, setOption] = useState('1');
  return (
    <Layout className="layout">
      <Header>
        <Menu theme="dark" mode="horizontal" defaultSelectedKeys={[option]} onClick={(e) => setOption(e.key)}>
          <Menu.Item key="1">Disaggregated Energy</Menu.Item>
          <Menu.Item key="2">Forecasted Energy</Menu.Item>
        </Menu>
      </Header>
      <Content style={{ padding: '0 50px' }}>
        <div className="site-layout-content">
          {
            renderIf(() => option === '1', () => <DisaggregatedEnergy />)
          }
          {
            renderIf(() => option === '2', () => <ForecastedEnergy />)
          }
        </div>
      </Content>
    </Layout>
  );
};

export default App;
