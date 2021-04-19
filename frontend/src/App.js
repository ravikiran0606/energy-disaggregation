import React, { useState } from 'react';
import { Layout, Menu } from 'antd';

import 'antd/dist/antd.css';
import MainDisaggregation from './MainDisaggregation';

const App = () => {
  const { Header, Content } = Layout;
  const [option, setOption] = useState('1');
  return (
    <Layout className="layout">
      <Header>
        <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['1']}>
          <Menu.Item key="1">Mains Energy</Menu.Item>
          <Menu.Item key="2">Disaggregated Energy</Menu.Item>
        </Menu>
      </Header>
      <Content style={{ padding: '0 50px' }}>
        <div className="site-layout-content">
          <MainDisaggregation />
        </div>
      </Content>
    </Layout>
  );
};

export default App;
