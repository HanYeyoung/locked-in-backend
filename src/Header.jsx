import React from 'react';

const Header = () => {
	return (
		<header className="border-b bg-black border-gray-500 font-{Helvetica} font-bold">
			<div className="max-w-7xl mx-auto">
				<div className="flex justify-between h-16 items-center">
					<div className="w-40 bg-transparent">
						<img src={require('./LastLockLogoDark.svg').default} />
					</div>
				</div>
			</div>
		</header>
	);
};

export default Header;